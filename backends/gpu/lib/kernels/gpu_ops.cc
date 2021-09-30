// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements MLIR operations for the cuda_ops library.

#include "tfrt/gpu/kernels/gpu_ops.h"

#include <iterator>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/miopen_wrapper.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/tensor/opdefs/tensor.h"

namespace tfrt {
namespace gpu {

//===----------------------------------------------------------------------===//
// GpuDialect Dialect
//===----------------------------------------------------------------------===//

GpuDialect::GpuDialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_gpu", context, TypeID::get<GpuDialect>()) {
  context->getOrLoadDialect<tfrt::t::TensorDialect>();
  allowUnknownTypes();
  allowUnknownOperations();

  addTypes<
#define GET_TYPEDEF_LIST
#include "tfrt/gpu/kernels/gpu_typedefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "tfrt/gpu/kernels/gpu_opdefs.cpp.inc"
      >();
}

namespace {
// Maps enum tag to CUDA and ROCm enum.
template <typename Tag>
struct EnumTraits;

template <>
struct EnumTraits<wrapper::DnnDataTypeTag> {
  using cuda_type = cudnnDataType_t;
  using rocm_type = miopenDataType_t;
};

template <>
struct EnumTraits<wrapper::BlasDataTypeTag> {
  using cuda_type = cudaDataType;
  using rocm_type = rocblas_datatype;
};

template <>
struct EnumTraits<wrapper::BlasComputeTypeTag> {
  using cuda_type = cublasComputeType_t;
  using rocm_type = rocblas_datatype;
};

template <>
struct EnumTraits<wrapper::BlasOperationTag> {
  using cuda_type = cublasOperation_t;
  using rocm_type = rocblas_operation;
};

template <>
struct EnumTraits<wrapper::BlasGemmAlgoTag> {
  using cuda_type = cublasGemmAlgo_t;
  using rocm_type = rocblas_gemm_algo;
};

template <>
struct EnumTraits<wrapper::BlasFillModeTag> {
  using cuda_type = cublasFillMode_t;
  using rocm_type = rocblas_fill;
};
}  // namespace

static Type GetType(MLIRContext *context, cudaDataType data_type) {
  switch (data_type) {
    case CUDA_R_16F:
      return Float16Type::get(context);
    case CUDA_R_32F:
      return Float32Type::get(context);
    case CUDA_R_64F:
      return Float64Type::get(context);
    default:
      llvm_unreachable("unexpected data type");
  }
}

static Type GetType(MLIRContext *context, cublasComputeType_t compute_type) {
  switch (compute_type) {
    case CUBLAS_COMPUTE_16F:
    case CUBLAS_COMPUTE_16F_PEDANTIC:
      return Float16Type::get(context);
    case CUBLAS_COMPUTE_32F:
    case CUBLAS_COMPUTE_32F_PEDANTIC:
    case CUBLAS_COMPUTE_32F_FAST_16F:
    case CUBLAS_COMPUTE_32F_FAST_16BF:
    case CUBLAS_COMPUTE_32F_FAST_TF32:
      return Float32Type::get(context);
    case CUBLAS_COMPUTE_64F:
    case CUBLAS_COMPUTE_64F_PEDANTIC:
      return Float64Type::get(context);
    case CUBLAS_COMPUTE_32I:
    case CUBLAS_COMPUTE_32I_PEDANTIC:
      return IntegerType::get(context, 32, IntegerType::Signed);
    default:
      llvm_unreachable("unexpected compute type");
  }
}

static Type GetType(MLIRContext *context, rocblas_datatype data_type) {
  switch (data_type) {
    case rocblas_datatype_f16_r:
      return Float16Type::get(context);
    case rocblas_datatype_f32_r:
      return Float32Type::get(context);
    case rocblas_datatype_f64_r:
      return Float64Type::get(context);
    default:
      llvm_unreachable("unexpected data type");
  }
}

template <typename Tag>
static Type GetType(EnumAttr<wrapper::Enum<Tag>> attribute) {
  MLIRContext *context = attribute.getContext();
  wrapper::Enum<Tag> value = attribute.getValue();
  switch (value.platform()) {
    case wrapper::Platform::CUDA:
      return GetType(context,
                     static_cast<typename EnumTraits<Tag>::cuda_type>(value));
    case wrapper::Platform::ROCm:
      return GetType(context,
                     static_cast<typename EnumTraits<Tag>::rocm_type>(value));
    default:
      llvm_unreachable("unexpected platform");
  }
}

template <typename T>
static ParseResult parseEnum(OpAsmParser &parser, EnumAttr<T> &attribute,
                             Expected<T> (*parse_func)(StringRef)) {
  StringRef name;
  if (failed(parser.parseKeyword(&name))) return failure();
  auto value = parse_func(name);
  if (!value) {
    return parser.emitError(parser.getCurrentLocation(),
                            toString(value.takeError()));
  }
  attribute = EnumAttr<T>::get(parser.getBuilder().getContext(), *value);
  return success();
}

template <typename T>
static ParseResult parseEnum(OpAsmParser &parser, EnumAttr<T> &attribute) {
  auto parse_func = [](StringRef name) { return wrapper::Parse<T>(name); };
  return parseEnum(parser, attribute, +parse_func);
}

template <typename Tag>
static ParseResult parseEnum(OpAsmParser &parser,
                             EnumAttr<wrapper::Enum<Tag>> &attribute) {
  auto parse_func = [](StringRef name) -> Expected<wrapper::Enum<Tag>> {
    auto cuda_value = wrapper::Parse<typename EnumTraits<Tag>::cuda_type>(name);
    if (cuda_value) return {*cuda_value};
    auto rocm_value = wrapper::Parse<typename EnumTraits<Tag>::rocm_type>(name);
    if (rocm_value) return {*rocm_value};
    return joinErrors(cuda_value.takeError(), rocm_value.takeError());
  };
  return parseEnum(parser, attribute, +parse_func);
}

// parseEnum overload also assigning the underlying type to one or more 'types'.
template <typename Tag, typename... Ts>
static ParseResult parseEnum(OpAsmParser &parser,
                             EnumAttr<wrapper::Enum<Tag>> &attribute,
                             Ts &...types) {
  if (failed(parseEnum(parser, attribute))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "unknown cudaDataType or rocblas_datatype enum");
  }

  if (auto type = GetType(attribute)) {
    Type dummy[]{(types = type)...};
    (void)dummy;
    return success();
  }

  return parser.emitError(
      parser.getCurrentLocation(),
      StrCat("could not infer type from ", attribute.getValue()));
}

template <typename T>
static void printEnum(OpAsmPrinter &printer, Operation *,
                      EnumAttr<T> attribute) {
  wrapper::operator<<(printer.getStream(), attribute.getValue());
}

template <typename Tag>
static void printEnum(OpAsmPrinter &printer, Operation *,
                      EnumAttr<wrapper::Enum<Tag>> attribute) {
  auto value = attribute.getValue();
  switch (value.platform()) {
    case wrapper::Platform::CUDA:
      wrapper::operator<<(
          printer.getStream(),
          static_cast<typename EnumTraits<Tag>::cuda_type>(value));
      break;
    case wrapper::Platform::ROCm:
      wrapper::operator<<(
          printer.getStream(),
          static_cast<typename EnumTraits<Tag>::rocm_type>(value));
      break;
    case wrapper::Platform::NONE:
      printer << value.platform();
      break;
  }
}

// printEnum overload ignoring one or more 'types'.
template <typename Tag, typename... Ts>
static void printEnum(OpAsmPrinter &printer, Operation *op,
                      EnumAttr<wrapper::Enum<Tag>> attribute, const Ts &...) {
  printEnum(printer, op, attribute);
}

static bool AllEqual(ArrayRef<wrapper::BlasDataType> types) {
  return llvm::all_of(types, [&](auto type) { return type == *types.begin(); });
}

static LogicalResult VerifyBlasSaxpyOp(BlasSaxpyOp op) {
  if (!AllEqual({op.typeAlpha(), op.typeX(), op.typeY(), op.executionType()})) {
    // The actual requirements of typeAlpha/typeX/typeY and executionType are
    // less strict than this, but at the moment we only use all float or all
    // double. Relax this check when we add support for e.g. mixed precision.
    return op.emitOpError(
        "typeAlpha, typeX, typeY and executionType need to match");
  }
  Type type_alpha = GetType(op.typeAlphaAttr());
  if (op.alpha().getType() != type_alpha) {
    return op.emitOpError("alpha's type doesn't match typeAlpha");
  }
  return mlir::success();
}

template <class OpTy>
static LogicalResult VerifyBlasGemmOp(OpTy op) {
  if (!AllEqual({op.typeA(), op.typeB(), op.typeC()})) {
    return op.emitOpError("typeA, typeB and typeC need to match");
  }
  Type compute_type = GetType(op.computeTypeAttr());
  if (op.alpha().getType() != compute_type ||
      op.beta().getType() != compute_type) {
    return op.emitOpError(
        "alpha's or beta's type is incompatible with computeType");
  }
  return mlir::success();
}

namespace conversion {

GpuConversionDialect::GpuConversionDialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_gpu_conversion", context,
              TypeID::get<GpuConversionDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/gpu/kernels/gpu_conversion_helper_opdefs.cpp.inc"
      >();
}

void AsyncExecuteOp::build(OpBuilder &builder, OperationState &result) {
  // Add a region with stream and chain block arguments.
  auto block = [&] {
    Region *region = result.addRegion();
    region->emplaceBlock();
    return region->begin();
  }();
  auto chain = block->addArgument(builder.getType<compiler::ChainType>());
  block->addArgument(builder.getType<StreamType>());

  // Return chain block argument.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&*block);
  builder.create<compiler::ReturnOp>(result.location, chain);
}

}  // namespace conversion

}  // namespace gpu
}  // namespace tfrt

// TableGen'd definitions
#define GET_TYPEDEF_CLASSES
#include "tfrt/gpu/kernels/gpu_typedefs.cpp.inc"
#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/gpu_opdefs.cpp.inc"
#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/gpu_conversion_helper_opdefs.cpp.inc"

Type tfrt::gpu::GpuDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  Type genType;
  if (succeeded(parser.parseKeyword(&typeTag)))
    generatedTypeParser(parser, typeTag, genType);
  return genType;
}

void tfrt::gpu::GpuDialect::printType(Type type,
                                      DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
