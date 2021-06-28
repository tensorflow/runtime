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
  using cuda_type = cublasDataType_t;
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
}  // namespace

template <typename T, typename F>
static ParseResult parseEnum(OpAsmParser &parser, EnumAttr<T> &attribute,
                             F &&parse_func) {
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
  return parseEnum(parser, attribute,
                   [](StringRef name) { return wrapper::Parse<T>(name); });
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
  return parseEnum(parser, attribute, parse_func);
}

template <typename T>
static void printEnum(OpAsmPrinter &printer, Operation *,
                      EnumAttr<T> attribute) {
  printer << attribute.getValue();
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

static Type GetBlasDataType(MLIRContext *context, cublasDataType_t data_type) {
  switch (data_type) {
    case CUDA_R_32F:
      return Float32Type::get(context);
    case CUDA_R_64F:
      return Float64Type::get(context);
    default:
      return {};
  }
}

static Type GetBlasDataType(MLIRContext *context, rocblas_datatype data_type) {
  switch (data_type) {
    case rocblas_datatype_f32_r:
      return Float32Type::get(context);
    case rocblas_datatype_f64_r:
      return Float64Type::get(context);
    default:
      return {};
  }
}

static Type GetBlasDataType(BlasDataTypeAttr attribute) {
  MLIRContext *context = attribute.getContext();
  wrapper::BlasDataType value = attribute.getValue();
  switch (value.platform()) {
    case wrapper::Platform::CUDA:
      return GetBlasDataType(context, static_cast<cublasDataType_t>(value));
    case wrapper::Platform::ROCm:
      return GetBlasDataType(context, static_cast<rocblas_datatype>(value));
    default:
      return {};
  }
}

template <typename... Ts>
static ParseResult parseBlasDataType(OpAsmParser &parser,
                                     BlasDataTypeAttr &attribute,
                                     Ts &...types) {
  if (failed(parseEnum(parser, attribute))) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "unknown cublasDataType_t or rocblas_datatype enum");
  }

  if (auto scale_type = GetBlasDataType(attribute)) {
    Type dummy[]{(types = scale_type)...};
    (void)dummy;
    return success();
  }

  return parser.emitError(
      parser.getCurrentLocation(),
      StrCat("could not infer type from ", attribute.getValue()));
}

template <typename... Ts>
static void printBlasDataType(OpAsmPrinter &printer, Operation *op,
                              BlasDataTypeAttr attribute, const Ts &...) {
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
  Type type_alpha = GetBlasDataType(op.typeAlphaAttr());
  if (op.alpha().getType() != type_alpha) {
    return op.emitOpError("alpha's type doesn't match typeAlpha");
  }
  return mlir::success();
}

template <class OpTy>
static LogicalResult VerifyBlasGemmOp(OpTy op) {
  if (!AllEqual({op.typeA(), op.typeB(), op.typeC(), op.computeType()})) {
    // The actual requirements of typeA/typeB/typeC and computeType are less
    // strict than this, but at the moment we only use all float or all double.
    // Relax this check when we add support for e.g. mixed precision.
    return op.emitOpError("typeA, typeB, typeC and computeType need to match");
  }
  Type compute_type = GetBlasDataType(op.computeTypeAttr());
  if (op.alpha().getType() != compute_type ||
      op.beta().getType() != compute_type) {
    return op.emitOpError("alpha's or beta's type don't match computeType");
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

mlir::OpFoldResult CastOp::fold(llvm::ArrayRef<mlir::Attribute>) {
  // Recursively search all defining cast ops for an operand of the result type.
  Type result_type = getResult().getType();
  SmallVector<Value, 2> operands = getOperands();
  while (!operands.empty()) {
    auto operand = operands.back();
    if (operand.getType() == result_type) return operand;
    operands.pop_back();
    if (auto cast_op = operand.getDefiningOp<conversion::CastOp>())
      copy(cast_op.getOperands(), std::back_inserter(operands));
  }
  return nullptr;
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
    generatedTypeParser(getContext(), parser, typeTag, genType);
  return genType;
}

void tfrt::gpu::GpuDialect::printType(Type type,
                                      DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
