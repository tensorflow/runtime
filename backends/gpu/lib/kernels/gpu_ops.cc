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
#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/cufft_wrapper.h"
#include "tfrt/gpu/wrapper/hipfft_wrapper.h"
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
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();

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

static Type GetType(MLIRContext *context, cudaDataType data_type) {
  switch (data_type) {
    case CUDA_R_16F:
      return Float16Type::get(context);
    case CUDA_R_32F:
      return Float32Type::get(context);
    case CUDA_R_64F:
      return Float64Type::get(context);
    case CUDA_C_32F:
      return ComplexType::get(Float32Type::get(context));
    default:
      llvm_unreachable("unexpected cuda data type");
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
      llvm_unreachable("unexpected cublas compute type");
  }
}

static Type GetType(MLIRContext *context, cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_HALF:
      return Float16Type::get(context);
    case CUDNN_DATA_FLOAT:
      return Float32Type::get(context);
    case CUDNN_DATA_DOUBLE:
      return Float64Type::get(context);
    default:
      llvm_unreachable("unexpected cudnn data type");
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
      llvm_unreachable("unexpected rocblas data type");
  }
}

static Type GetType(MLIRContext *context, miopenDataType_t data_type) {
  switch (data_type) {
    case miopenHalf:
      return Float16Type::get(context);
    case miopenFloat:
      return Float32Type::get(context);
    default:
      llvm_unreachable("unexpected miopen data type");
  }
}

template <typename T>
static Type GetType(EnumAttr<T> attribute) {
  MLIRContext *context = attribute.getContext();
  T value = attribute.getValue();
  switch (value.platform()) {
    case wrapper::Platform::CUDA:
      using CudaT = typename T::template PlatformType<wrapper::Platform::CUDA>;
      return GetType(context, static_cast<CudaT>(value));
    case wrapper::Platform::ROCm:
      using RocmT = typename T::template PlatformType<wrapper::Platform::ROCm>;
      return GetType(context, static_cast<RocmT>(value));
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
  return parseEnum(parser, attribute, &T::Parse);
}

// Overload for wrapper::Platform.
static ParseResult parseEnum(OpAsmParser &parser,
                             EnumAttr<wrapper::Platform> &attribute) {
  return parseEnum(parser, attribute, &wrapper::ParsePlatform);
}

template <typename T>
static ParseResult parseConvAlgoEnum(OpAsmParser &parser,
                                     EnumAttr<T> &attribute) {
  MLIRContext *context = parser.getBuilder().getContext();
  uint64_t rocm_value;
  auto result = parser.parseOptionalInteger(rocm_value);
  if (!result.has_value() || result.value())
    return parseEnum<T>(parser, attribute);
  attribute = EnumAttr<T>::get(context, T(rocm_value, wrapper::Platform::ROCm));
  return success();
}

// Overloads for DnnConv*AlgoAttr.
static ParseResult parseEnum(OpAsmParser &parser,
                             DnnConvFwdAlgoAttr &attribute) {
  return parseConvAlgoEnum(parser, attribute);
}
static ParseResult parseEnum(OpAsmParser &parser,
                             DnnConvBwdDataAlgoAttr &attribute) {
  return parseConvAlgoEnum(parser, attribute);
}
static ParseResult parseEnum(OpAsmParser &parser,
                             DnnConvBwdFilterAlgoAttr &attribute) {
  return parseConvAlgoEnum(parser, attribute);
}

// parseEnum overload also assigning the underlying type to one or more 'types'.
template <typename T, typename... Types>
static ParseResult parseEnum(OpAsmParser &parser, EnumAttr<T> &attribute,
                             Types &...types) {
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

template <typename... Types>
static ParseResult parseTypeAttr(OpAsmParser &parser, TypeAttr &attribute,
                                 Types &...types) {
  Type type;
  if (failed(parser.parseType(type))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to parse type");
  }

  attribute = TypeAttr::get(type);
  Type dummy[]{(types = type)...};
  (void)dummy;
  return success();
}

template <typename T>
static void printEnum(OpAsmPrinter &printer, Operation *,
                      const EnumAttr<T> &attribute) {
  printer << attribute.getValue();
}

// printEnum overload ignoring one or more 'types'.
template <typename T, typename... Types>
static void printEnum(OpAsmPrinter &printer, Operation *op,
                      const EnumAttr<T> &attribute, const Types &...) {
  printEnum(printer, op, attribute);
}

template <typename... Types>
static void printTypeAttr(OpAsmPrinter &printer, Operation *op,
                          const TypeAttr &attribute, const Types &...) {
  printer.printType(attribute.getValue());
}

static bool AllEqual(ArrayRef<wrapper::BlasDataType> types) {
  return llvm::all_of(types, [&](auto type) { return type == *types.begin(); });
}

LogicalResult BlasSaxpyOp::verify() {
  BlasSaxpyOp op = *this;
  if (!AllEqual({op.getTypeAlpha(), op.getTypeX(), op.getTypeY(),
                 op.getExecutionType()})) {
    // The actual requirements of typeAlpha/typeX/typeY and executionType are
    // less strict than this, but at the moment we only use all float or all
    // double. Relax this check when we add support for e.g. mixed precision.
    return op.emitOpError(
        "typeAlpha, typeX, typeY and executionType need to match");
  }
  Type type_alpha = GetType(op.getTypeAlphaAttr());
  if (op.getAlpha().getType() != type_alpha) {
    return op.emitOpError("alpha's type doesn't match typeAlpha");
  }
  return mlir::success();
}

template <class OpTy>
static LogicalResult VerifyBlasGemmOp(OpTy op) {
  if (op.getTypeA() != op.getTypeB()) {
    return op.emitOpError("typeA and typeB need to match");
  }
  return mlir::success();
}

LogicalResult BlasGemmOp::verify() { return VerifyBlasGemmOp(*this); }
LogicalResult BlasGemmBatchExOp::verify() { return VerifyBlasGemmOp(*this); }

LogicalResult BlasScalOp::verify() {
  BlasScalOp op = *this;
  if (!AllEqual({op.getTypeAlpha(), op.getTypeX(), op.getExecutionType()})) {
    // The actual requirements of typeAlpha/typeX/executionType are less strict
    // than this, but at the moment we only use all float or all double. Relax
    // this check when we add support for e.g. mixed precision.
    return op.emitOpError("typeAlpha, typeX, and executionType need to match");
  }
  Type type_alpha = GetType(op.getTypeAlphaAttr());
  if (op.getAlpha().getType() != type_alpha) {
    return op.emitOpError("alpha's type doesn't match typeAlpha");
  }
  return mlir::success();
}

LogicalResult FftCreateOp::verify() {
  if (getDims().empty() || getDims().size() > 3)
    return emitOpError("dims should have rank 1, 2, or 3.");
  if (getInStrides().size() != getDims().size() + 1)
    return emitOpError("in_strides should be one larger than dims.");
  if (getOutStrides().size() != getDims().size() + 1)
    return emitOpError("out_strides should be one larger than dims.");
  return mlir::success();
}

void StreamifyOp::build(OpBuilder &builder, OperationState &state,
                        ValueRange results) {
  state.addTypes(TypeRange(results));

  // Add a region with stream and chain block arguments.
  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  auto chain =
      block.addArgument(builder.getType<compiler::ChainType>(), state.location);
  block.addArgument(builder.getType<StreamType>(), state.location);

  // Add return of chain block argument and provided results.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);
  builder.create<compiler::ReturnOp>(state.location, chain)
      ->insertOperands(1, results);
}

// Copied from MLIR's GPUDialect.cpp.
static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<mlir::gpu::AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

ParseResult StreamifyOp::parse(OpAsmParser &parser, OperationState &result) {
  Type token_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> async_dependencies;
  if (parseAsyncDependencies(parser, token_type, async_dependencies))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto body = std::make_unique<Region>();
  if (parser.parseRegion(*body)) return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);
  result.addRegion(std::move(body));

  if (succeeded(parser.parseOptionalColon()) &&
      parser.parseTypeList(result.types))
    return failure();
  if (token_type) result.addTypes(token_type);

  auto context = parser.getBuilder().getContext();
  return parser.resolveOperands(async_dependencies,
                                mlir::gpu::AsyncTokenType::get(context),
                                result.operands);
}

void StreamifyOp::print(OpAsmPrinter &printer) {
  if (asyncToken()) printer << " async";
  if (!getAsyncDependencies().empty()) {
    printer << " [";
    llvm::interleaveComma(getAsyncDependencies(), printer);
    printer << "]";
  }

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printer << ' ';
  printer.printRegion(getBody());

  if (!results().empty()) printer << " : " << results().getTypes();
}

Operation::result_range StreamifyOp::results() {
  auto results = getOperation()->getResults();
  if (asyncToken()) return results.drop_back();
  return results;
}

OpResult StreamifyOp::asyncToken() {
  if (getOperation()->getNumResults() == 0) return nullptr;
  OpResult result = getOperation()->getResults().back();
  if (!result.getType().isa<mlir::gpu::AsyncTokenType>()) return nullptr;
  return result;
}

LogicalResult StreamifyOp::verify() {
  TypeRange return_types =
      SingleBlock::getBody()->getTerminator()->getOperandTypes();
  if (return_types.empty() || !return_types.front().isa<compiler::ChainType>())
    return emitOpError("first return operand type is not a !tfrt.chain");
  if (return_types.drop_front() != TypeRange(results()))
    return emitOpError("trailing return types don't match result types");
  return mlir::success();
}

}  // namespace gpu
}  // namespace tfrt

// TableGen'd definitions
#define GET_TYPEDEF_CLASSES
#include "tfrt/gpu/kernels/gpu_typedefs.cpp.inc"
#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/gpu_opdefs.cpp.inc"

Type tfrt::gpu::GpuDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  Type genType;
  if (generatedTypeParser(parser, &typeTag, genType).has_value())
    return genType;
  auto identifier = StringAttr::get(parser.getContext(), getDialectNamespace());
  return OpaqueType::get(identifier, typeTag);
}

void tfrt::gpu::GpuDialect::printType(Type type,
                                      DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
