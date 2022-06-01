/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iterator>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/jitrt/conversion/custom_call_to_llvm.h"
#include "tfrt/jitrt/conversion/rt_passes.h"
#include "tfrt/jitrt/custom_call.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"

namespace tfrt {
namespace jitrt {
namespace {

using llvm::DenseMap;

using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::ComplexType;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DenseIntOrFPElementsAttr;
using mlir::failure;
using mlir::FailureOr;
using mlir::FunctionType;
using mlir::getStridesAndOffset;
using mlir::ImplicitLocOpBuilder;
using mlir::IntegerType;
using mlir::LLVMTypeConverter;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefDescriptor;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::OperationPass;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::TypeConverter;
using mlir::TypeID;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantOp;
using mlir::func::CallOp;
using mlir::func::FuncOp;

namespace LLVM = mlir::LLVM;

#define GEN_PASS_CLASSES
#include "tfrt/jitrt/conversion/rt_gen_passes.h.inc"

//===----------------------------------------------------------------------===//
// Runtime C API declaration (see runtime.h header file).
//===----------------------------------------------------------------------===//

static constexpr const char *kGetResultStorage = "runtimeGetResultStorage";
static constexpr const char *kSetError = "runtimeSetError";
static constexpr const char *kCustomCall = "runtimeCustomCall";

struct RuntimeAPI {
  static LLVM::LLVMPointerType OpaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMPointerType CustomCallArgumentsType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static LLVM::LLVMPointerType CustomCallAttributesType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static FunctionType GetResultStorageFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto i64 = IntegerType::get(ctx, 64);
    auto storage = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, i64}, {storage});
  }

  static FunctionType SetErrorFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto error_msg = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, error_msg}, {});
  }

  static FunctionType CustomCallFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto callee = OpaquePointerType(ctx);
    auto args = CustomCallArgumentsType(ctx);
    auto attrs = CustomCallAttributesType(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {kernel_context, callee, args, attrs}, {i1});
  }

  static FunctionType DirectCustomCallFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto args = CustomCallArgumentsType(ctx);
    auto attrs = CustomCallAttributesType(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {kernel_context, args, attrs}, {i1});
  }
};

// Adds function declaration if it doesn't already exist.
static void AddDeclaration(ModuleOp module, StringRef name, FunctionType type) {
  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
  if (module.lookupSymbol(name)) return;

  MLIRContext *ctx = module.getContext();
  FuncOp func = b.create<FuncOp>(name, type);
  func.setPrivate();

  // TODO(ezhulenev): Add per-argument nocapture attributes?
  func->setAttr("passthrough",
                ArrayAttr::get(ctx, {StringAttr::get(ctx, "nounwind")}));
}

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(ModuleOp module) {
  auto add = [&](StringRef name, FunctionType type) {
    AddDeclaration(module, name, type);
  };

  MLIRContext *ctx = module.getContext();
  add(kGetResultStorage, RuntimeAPI::GetResultStorageFunctionType(ctx));
  add(kSetError, RuntimeAPI::SetErrorFunctionType(ctx));
  add(kCustomCall, RuntimeAPI::CustomCallFunctionType(ctx));
}

// -------------------------------------------------------------------------- //

class RuntimeTypeConverter : public TypeConverter {
 public:
  RuntimeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(ConvertKernelContextType);
    addConversion(ConvertStatusType);
  }

  static llvm::Optional<Type> ConvertKernelContextType(KernelContextType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  }

  static llvm::Optional<Type> ConvertStatusType(StatusType type) {
    return IntegerType::get(type.getContext(), 1);
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.set_output to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetOutputOpLowering : public OpConversionPattern<SetOutputOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetOutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto kernel_context = adaptor.ctx();
    auto index = rewriter.create<ConstantOp>(loc, adaptor.indexAttr());

    // Get a pointer to the result value storage from the runtime.
    auto result_ptr_ty = RuntimeAPI::OpaquePointerType(rewriter.getContext());
    auto result_ptr = rewriter.create<CallOp>(
        loc, kGetResultStorage, TypeRange(result_ptr_ty),
        ValueRange({kernel_context, index}));

    // Cast from i8* to the LLVM pointer type to store the result.
    auto stored_type = getTypeConverter()->convertType(op.value().getType());
    if (!stored_type)
      return rewriter.notifyMatchFailure(
          op, "failed to convert output type to LLVM type");

    auto casted_result_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(stored_type), result_ptr.getResult(0));

    // Store the output value into the result value storage.
    auto value = adaptor.value();
    rewriter.create<LLVM::StoreOp>(loc, value, casted_result_ptr.getResult());

    // Erase the original runtime operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.is_ok to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class IsOkOpLowering : public OpConversionPattern<IsOkOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IsOkOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Just pass through the converted operand.
    rewriter.replaceOp(op, adaptor.status());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.custom_call to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

// Helper functions for encoding attributes and values for custom calls.

static bool IsSupportedScalarType(Type type) {
  auto is_supported_width = [](unsigned width, ArrayRef<unsigned> supported) {
    return llvm::any_of(supported, [&](unsigned w) { return w == width; });
  };

  if (auto integer = type.dyn_cast<mlir::IntegerType>())
    return is_supported_width(integer.getWidth(), {1, 32, 64});

  if (auto fp = type.dyn_cast<mlir::FloatType>())
    return is_supported_width(fp.getWidth(), {32, 64});

  return false;
}

static bool IsSupportedShapedType(ShapedType shape) {
  return shape.getRank() == 1 && IsSupportedScalarType(shape.getElementType());
  return false;
}

static TypeID ScalarRuntimeTypeId(Type type) {
  if (type.isUnsignedInteger(8)) return TypeID::get<Tagged<uint8_t>>();
  if (type.isUnsignedInteger(32)) return TypeID::get<Tagged<uint32_t>>();
  if (type.isUnsignedInteger(64)) return TypeID::get<Tagged<uint64_t>>();

  if (type.isInteger(1)) return TypeID::get<Tagged<bool>>();
  if (type.isInteger(32)) return TypeID::get<Tagged<int32_t>>();
  if (type.isInteger(64)) return TypeID::get<Tagged<int64_t>>();

  if (type.isF32()) return TypeID::get<Tagged<float>>();
  if (type.isF64()) return TypeID::get<Tagged<double>>();

  assert(false && "unsupported type id");
  return TypeID::getFromOpaquePointer(reinterpret_cast<void *>(0xDEADBEEF));
}

static DType ScalarDType(Type type) {
  // Unsigned integer types.
  if (type.isUnsignedInteger(8)) return DType::UI8;
  if (type.isUnsignedInteger(16)) return DType::UI16;
  if (type.isUnsignedInteger(32)) return DType::UI32;
  if (type.isUnsignedInteger(64)) return DType::UI64;

  // Signed integer types.
  if (type.isInteger(1)) return DType::I1;
  if (type.isInteger(8)) return DType::I8;
  if (type.isInteger(16)) return DType::I16;
  if (type.isInteger(32)) return DType::I32;
  if (type.isInteger(64)) return DType::I64;

  // Floating point types.
  if (type.isF16()) return DType::F16;
  if (type.isF32()) return DType::F32;
  if (type.isF64()) return DType::F64;
  if (type.isBF16()) return DType::BF16;

  // Complex types.
  if (auto complex = type.dyn_cast<ComplexType>()) {
    if (complex.getElementType().isF32()) return DType::Complex64;
    if (complex.getElementType().isF64()) return DType::Complex128;
  }

  assert(false && "unsupported type id");
  return DType::Invalid;
}

static TypeID ArrayRuntimeTypeId(Type shaped) {
  auto type = shaped.cast<ShapedType>().getElementType();
  assert(shaped.cast<ShapedType>().getRank() == 1 && "unsupported rank");

  if (type.isInteger(32)) return TypeID::get<Tagged<ArrayRef<int32_t>>>();
  if (type.isInteger(64)) return TypeID::get<Tagged<ArrayRef<int64_t>>>();
  if (type.isF32()) return TypeID::get<Tagged<ArrayRef<float>>>();
  if (type.isF64()) return TypeID::get<Tagged<ArrayRef<double>>>();

  assert(false && "unsupported type id");
  return TypeID::getFromOpaquePointer(reinterpret_cast<void *>(0xDEADBEEF));
}

// Packs scalar attribute as a global constant. Returns `!llvm.ptr<AttrType>`.
static Value PackScalarAttribute(Globals &g, ImplicitLocOpBuilder &b,
                                 Attribute value, StringRef symbol_base) {
  auto global = g.GetOrCreate(b, value, symbol_base);
  return Globals::AddrOf(b, global);
}

// Packs TypeID as `i64` constant value and casts it to the `!llvm.ptr<i8>`,
// because type id internally is implemented as an opaque pointer.
static Value PackTypeId(Globals &g, ImplicitLocOpBuilder &b, TypeID type_id) {
  auto i64 = reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer());
  auto cst = b.create<ConstantOp>(b.getI64IntegerAttr(i64));
  auto ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  return b.create<LLVM::IntToPtrOp>(ptr, cst);
}

// Packs string as a module global constants. Returns `!llvm.ptr<EncodedStr>`.
// We always pass string with the size to the runtime intrinsics, because
// computing the length of null-terminated string can be expensive, and we need
// it to construct llvm::StringRef at run time.
static Value PackString(Globals &g, ImplicitLocOpBuilder &b, StringRef strref,
                        StringRef symbol_base) {
  MLIRContext *ctx = b.getContext();
  int64_t size = strref.size();

  // Encoded string type: !llvm.struct<(i64, !llvm.ptr<array<i8 x len>>)>.
  Type arr = LLVM::LLVMArrayType::get(b.getI8Type(), 1 + size);
  Type ptr = LLVM::LLVMPointerType::get(arr);
  Type type = LLVM::LLVMStructType::getLiteral(ctx, {b.getI64Type(), ptr});

  // Global constant initializer for the encoded string structure
  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    // String size and pointer to a null-terminated string.
    Value num_elements = ib.create<ConstantOp>(ib.getI64IntegerAttr(size));
    Value str = Globals::AddrOf(ib, g.GetOrCreate(b, strref, "__rt_str"));

    // Store size and pointer into the struct.
    Value encoded = ib.create<LLVM::UndefOp>(type);
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, num_elements,
                                             ib.getI64ArrayAttr(0));
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, str,
                                             ib.getI64ArrayAttr(1));
    ib.create<LLVM::ReturnOp>(encoded);
  };

  auto value = b.getStringAttr(strref);
  auto global = g.GetOrCreate(b, value, type, symbol_base, init);
  return Globals::AddrOf(b, global);
}

// Packs array attribute as a global constant. Returns `!llvm.ptr<EncodedArr>`.
static Value PackArrayAttribute(Globals &g, ImplicitLocOpBuilder &b,
                                Attribute value, StringRef symbol_base) {
  MLIRContext *ctx = b.getContext();

  // We only support dense attributes for now.
  DenseIntOrFPElementsAttr dense = value.cast<DenseIntOrFPElementsAttr>();
  int64_t size = dense.getNumElements();

  // Encoded array type: !llvm.struct<(i64, !llvm.array<element_type x size>)>.
  Type element_type = dense.getElementType();
  Type arr = LLVM::LLVMArrayType::get(element_type, size);
  Type type = LLVM::LLVMStructType::getLiteral(ctx, {b.getI64Type(), arr});

  // Global constant initializer for the encoded array structure
  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    // Array size and values.
    Value num_elements = ib.create<ConstantOp>(b.getI64IntegerAttr(size));
    Value values = b.create<LLVM::ConstantOp>(arr, dense);

    // Store size and values into the struct.
    Value encoded = ib.create<LLVM::UndefOp>(type);
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, num_elements,
                                             ib.getI64ArrayAttr(0));
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, values,
                                             ib.getI64ArrayAttr(1));

    ib.create<LLVM::ReturnOp>(encoded);
  };

  auto global = g.GetOrCreate(b, value, type, symbol_base, init);
  return Globals::AddrOf(b, global);
}

// Packs value on the stack. Returns `!llvm.ptr<ValueType>`.
static Value PackValue(ImplicitLocOpBuilder &b, Value value) {
  Type ptr = LLVM::LLVMPointerType::get(value.getType());
  Value one = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(ptr, one, 0);
  b.create<LLVM::StoreOp>(value, mem);

  return mem;
}

// -------------------------------------------------------------------------- //
// Custom call attributes encoding.
// -------------------------------------------------------------------------- //

struct StringAttrEncoding : public CustomCallAttrEncoding {
  LogicalResult Match(StringRef name, Attribute attr) const final {
    return success(attr.isa<StringAttr>());
  }

  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute attr) const override {
    auto str = attr.cast<StringAttr>();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, TypeID::get<Tagged<llvm::StringRef>>());
    encoded.value = PackString(g, b, str, kAttrValue);
    return encoded;
  }
};

struct ScalarAttrEncoding : public CustomCallAttrEncoding {
  LogicalResult Match(StringRef name, Attribute attr) const final {
    return success(IsSupportedScalarType(attr.getType()));
  }

  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute attr) const override {
    Type type = attr.getType();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
    encoded.value = PackScalarAttribute(g, b, attr, kAttrValue);

    return encoded;
  }
};

struct ArrayAttrEncoding : public CustomCallAttrEncoding {
  LogicalResult Match(StringRef name, Attribute attr) const final {
    if (auto dense = attr.dyn_cast<DenseIntOrFPElementsAttr>())
      return success(IsSupportedShapedType(dense.getType()));
    return failure();
  }

  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute attr) const override {
    ShapedType type = attr.getType().cast<ShapedType>();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, ArrayRuntimeTypeId(type));
    encoded.value = PackArrayAttribute(g, b, attr, kAttrValue);

    return encoded;
  }
};

// -------------------------------------------------------------------------- //
// Custom call arguments encodings.
// -------------------------------------------------------------------------- //

// Encodes scalar operands.
class ScalarArgEncoding : public CustomCallArgEncoding {
 public:
  LogicalResult Match(Value value, Value converted) const final {
    return success(IsSupportedScalarType(value.getType()));
  }

  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, Value value,
                            Value converted) const final {
    Type type = converted.getType();

    Encoded encoded;
    encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
    encoded.value = PackValue(b, converted);

    return encoded;
  }
};

// Encodes MemRef operands according to the (Strided)MemrefView ABI.
class MemrefArgEncoding : public CustomCallArgEncoding {
 public:
  LogicalResult Match(Value value, Value converted) const final {
    return success(value.getType().isa<MemRefType>());
  }

  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, Value value,
                            Value converted) const override {
    auto memref_type = value.getType().cast<MemRefType>();

    // If memref has non-identity layout we use `StridedMemrefView` to
    // distinguish it from the default row-major memref.
    auto type_id = memref_type.getLayout().isIdentity()
                       ? TypeID::get<Tagged<MemrefView>>()
                       : TypeID::get<Tagged<StridedMemrefView>>();

    Encoded encoded;
    encoded.type_id = PackTypeId(g, b, type_id);
    encoded.value = PackValue(b, EncodeMemRef(b, memref_type, converted));

    return encoded;
  }

 private:
  // Encodes memref as LLVM struct value:
  //
  //   { i8: dtype, i8: rank, ptr<i8>: data,
  //     array<2*rank x i64>: sizes_and_strides }
  //
  // This is a type erased version of the MLIR memref descriptor without base
  // pointer. We pack sizes and strides as a single array member, so that on
  // the runtime side we can read it back using C flexible array member.
  Value EncodeMemRef(ImplicitLocOpBuilder &b, MemRefType memref_ty,
                     Value descriptor) const {
    MLIRContext *ctx = b.getContext();
    Location loc = b.getLoc();

    // Encode sizes together with strides as a single array.
    int64_t sizes_and_strides_size = 2 * memref_ty.getRank();

    // Encoded memref type: !llvm.struct<(i8, i8, ptr<i8>, array<... x i64>)>.
    Type i8 = b.getI8Type();
    Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
    Type arr = LLVM::LLVMArrayType::get(b.getI64Type(), sizes_and_strides_size);
    Type type = LLVM::LLVMStructType::getLiteral(ctx, {i8, i8, ptr, arr});

    // Helper to unpack MLIR strided memref descriptor value.
    MemRefDescriptor desc(descriptor);

    DType element_dtype = ScalarDType(memref_ty.getElementType());

    // Create values for filling encoded memref struct.
    Value dtype = b.create<ConstantOp>(
        b.getI8IntegerAttr(static_cast<uint8_t>(element_dtype)));
    Value rank = b.create<ConstantOp>(b.getI8IntegerAttr(memref_ty.getRank()));
    Value data = b.create<LLVM::BitcastOp>(ptr, desc.alignedPtr(b, loc));

    auto offset = [&](int64_t i) { return b.getI64ArrayAttr(i); };
    auto i64 = [&](int64_t i) { return b.getI64IntegerAttr(i); };

    // Get the statically known strides and offset from the memref type.
    llvm::SmallVector<int64_t> strides;
    int64_t memref_offset;
    if (failed(getStridesAndOffset(memref_ty, strides, memref_offset)))
      strides.resize(memref_ty.getRank(), ShapedType::kDynamicStrideOrOffset);

    // Build encoded memref sizes + strides: !llvm.array<... x i64>
    Value payload = b.create<LLVM::UndefOp>(arr);
    for (unsigned i = 0; i < memref_ty.getRank(); ++i) {
      int64_t dim_size = memref_ty.getDimSize(i);
      int64_t stride_size = strides[i];

      Value dim = ShapedType::isDynamic(dim_size)
                      ? desc.size(b, loc, i)
                      : b.create<ConstantOp>(i64(dim_size));

      Value stride = ShapedType::isDynamic(stride_size)
                         ? desc.stride(b, loc, i)
                         : b.create<ConstantOp>(i64(stride_size));

      auto size_pos = offset(i);
      auto stride_pos = offset(memref_ty.getRank() + i);

      payload = b.create<LLVM::InsertValueOp>(arr, payload, dim, size_pos);
      payload = b.create<LLVM::InsertValueOp>(arr, payload, stride, stride_pos);
    }

    // Construct encoded memref value.
    Value memref = b.create<LLVM::UndefOp>(type);
    memref = b.create<LLVM::InsertValueOp>(type, memref, dtype, offset(0));
    memref = b.create<LLVM::InsertValueOp>(type, memref, rank, offset(1));
    memref = b.create<LLVM::InsertValueOp>(type, memref, data, offset(2));
    memref = b.create<LLVM::InsertValueOp>(type, memref, payload, offset(3));

    return memref;
  }
};

// ------------------------------------------------------------------------- -//

static FailureOr<Value> EncodeArguments(
    CustomCallArgEncodingSet &encodings, Globals &g,
    DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args,
    ImplicitLocOpBuilder &b, ValueRange operands, ValueRange converted) {
  llvm::SmallVector<CustomCallArgEncoding::Encoded> encoded;

  // Encode all arguments as a set of pointers (skip the kernel context).
  for (auto tuple : llvm::drop_begin(llvm::zip(operands, converted))) {
    // Check if the value was already encoded.
    auto it = encoded_args.find(std::get<0>(tuple));
    if (it != encoded_args.end()) {
      encoded.push_back(it->second);
      continue;
    }

    // Otherwise encode it right after the converted value definition.
    OpBuilder::InsertionGuard guard(b);
    if (auto *defining_op = std::get<1>(tuple).getDefiningOp()) {
      b.setInsertionPointAfter(defining_op);
    } else {
      b.setInsertionPointToStart(std::get<1>(tuple).getParentBlock());
    }

    auto encoded_arg =
        encodings.Encode(g, b, std::get<0>(tuple), std::get<1>(tuple));
    if (failed(encoded_arg)) return failure();
    encoded.push_back(*encoded_arg);
    encoded_args.try_emplace(std::get<0>(tuple), *encoded_arg);
  }

  // We store encoded arguments as `!llvm.array<ptr<i8> x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  Type type = LLVM::LLVMArrayType::get(ptr, 1 + encoded.size() * 2);

  // Prepare an array for encoding arguments.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    Value bcasted = b.createOrFold<LLVM::BitcastOp>(ptr, value);
    arr = b.create<LLVM::InsertValueOp>(type, arr, bcasted,
                                        b.getI64ArrayAttr(offset));
  };

  // Insert the number of encoded arguments.
  Attribute num_args = b.getI64IntegerAttr(encoded.size());
  insert_value(PackScalarAttribute(g, b, num_args, "__rt_num_args"), 0);

  // Store encoded arguments into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallArgEncoding::Encoded encoded = pair.value();
    int64_t offset = 1 + pair.index() * 2;

    insert_value(encoded.type_id, offset + 0);
    insert_value(encoded.value, offset + 1);
  }

  // Store constructed arguments array on the stack and return a pointer to it.
  Value c1 = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(LLVM::LLVMPointerType::get(type), c1, 0);
  b.create<LLVM::StoreOp>(arr, mem);

  // Return a pointer to the first element of the arguments array.
  Type ptr_ptr = mlir::LLVM::LLVMPointerType::get(ptr);
  Value c0 = b.create<ConstantOp>(b.getI64IntegerAttr(0));
  Value gep = b.create<LLVM::GEPOp>(ptr_ptr, mem, ValueRange({c0, c0}));
  return gep;
}

// Encodes attributes into the global constant (array of pointers to the
// attributes data, which are also stored as global constants).
static FailureOr<Value> EncodeAttributes(CustomCallAttrEncodingSet &encodings,
                                         Globals &g, ImplicitLocOpBuilder &b,
                                         ArrayRef<NamedAttribute> attrs) {
  using EncodedAttr = std::pair<StringRef, CustomCallAttrEncoding::Encoded>;

  // Skip attributes passed explicitly as a custom call argument.
  auto skip = [](NamedAttribute attr) {
    return attr.getName() == "callee" || attr.getName() == "direct";
  };

  // In addition to encoded attribues we encode the number of attributes.
  int64_t n_attrs = attrs.size() - llvm::count_if(attrs, skip);

  // We store encoded attributes as `!llvm.array<ptr<i8> x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  Type type = LLVM::LLVMArrayType::get(ptr, 1 + n_attrs * 3);

  // Prepare a global constant for storing encoded attributes.
  LLVM::GlobalOp global = [&]() {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(g.module().getBody());
    return b.create<LLVM::GlobalOp>(
        type, /*isConstant=*/true, LLVM::Linkage::Internal,
        g.UniqueSymName("__rt_custom_call_attrs"), Attribute());
  }();

  // Get a pointer to the first element of the array: !llvm.ptr<ptr<i8>>.
  Type ptr_ptr = mlir::LLVM::LLVMPointerType::get(ptr);
  Value c0 = b.create<ConstantOp>(b.getI64IntegerAttr(0));
  Value addr = Globals::AddrOf(b, global);
  Value gep = b.create<LLVM::GEPOp>(ptr_ptr, addr, ValueRange({c0, c0}));

  OpBuilder::InsertionGuard guard(b);

  // Create a global constant initializer block.
  mlir::Region &region = global.getInitializerRegion();
  mlir::Block *block = b.createBlock(&region);

  // Build attributes encoding inside the initializer block.
  b.setInsertionPointToStart(block);

  llvm::SmallVector<EncodedAttr> encoded;
  for (auto &attr : attrs) {
    if (skip(attr)) continue;

    // Try to encode the attribute as a set of pointers.
    auto encoded_attr = encodings.Encode(g, b, attr.getName(), attr.getValue());
    if (failed(encoded_attr)) return failure();
    encoded.emplace_back(attr.getName(), *encoded_attr);
  }

  // Sort encoded attributes in lexicographical order.
  llvm::sort(encoded, [](auto &a, auto &b) { return a.first < b.first; });

  // Prepare an array for encoding attributes.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    Value bcasted = b.createOrFold<LLVM::BitcastOp>(ptr, value);
    arr = b.create<LLVM::InsertValueOp>(type, arr, bcasted,
                                        b.getI64ArrayAttr(offset));
  };

  // Insert the number of encoded attributes.
  Attribute num_attrs = b.getI64IntegerAttr(n_attrs);
  insert_value(PackScalarAttribute(g, b, num_attrs, "__rt_num_attrs"), 0);

  // Insert encoded attributes into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallAttrEncoding::Encoded encoded = pair.value().second;
    int64_t offset = 1 + pair.index() * 3;

    insert_value(encoded.name, offset + 0);
    insert_value(encoded.type_id, offset + 1);
    insert_value(encoded.value, offset + 2);
  }

  // Return attributes array from the global initializer block.
  b.create<LLVM::ReturnOp>(arr);

  // Return a pointer to the encoded attributes: `!llvm.ptr<ptr<i8>>` (void**).
  return gep;
}

class CustomCallOpLowering : public OpConversionPattern<CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  CustomCallOpLowering(
      TypeConverter &converter, MLIRContext *ctx, Globals &globals,
      CustomCallArgEncodingSet &arg_encoding,
      CustomCallAttrEncodingSet &attr_encoding,
      DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args)
      : OpConversionPattern(converter, ctx),
        globals_(globals),
        arg_encoding_(arg_encoding),
        attr_encoding_(attr_encoding),
        encoded_args_(encoded_args) {}

  LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Encode operation arguments as a runtime API arguments.
    auto args = EncodeArguments(arg_encoding_, globals_, encoded_args_, b,
                                op->getOperands(), adaptor.getOperands());
    if (failed(args)) return op.emitOpError() << "failed to encode arguments";

    // Encode operation attributes as a runtime API argument.
    auto attrs = EncodeAttributes(attr_encoding_, globals_, b, op->getAttrs());
    if (failed(attrs)) return op.emitOpError() << "failed to encode attributes";

    if (op.direct()) {
      // Call custom call target directly.
      auto type = RuntimeAPI::DirectCustomCallFunctionType(op.getContext());
      AddDeclaration(op->getParentOfType<ModuleOp>(), op.callee(), type);

      rewriter.replaceOpWithNewOp<CallOp>(
          op, op.callee(), TypeRange(rewriter.getI1Type()),
          ValueRange({adaptor.ctx(), *args, *attrs}));

    } else {
      // Otherwise pass the custom call callee to the generic custom call API.
      auto callee = Globals::OpaqueAddrOf(
          b, globals_.GetOrCreate(b, op.callee(), "__rt_custom_call_callee"));

      // Call runtime API to call the custom call target.
      rewriter.replaceOpWithNewOp<CallOp>(
          op, kCustomCall, TypeRange(rewriter.getI1Type()),
          ValueRange({adaptor.ctx(), callee, *args, *attrs}));
    }

    return success();
  }

 private:
  Globals &globals_;
  CustomCallArgEncodingSet &arg_encoding_;
  CustomCallAttrEncodingSet &attr_encoding_;
  DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args_;
};

//===----------------------------------------------------------------------===//
// Convert rt.set_error to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetErrorOpLowering : public OpConversionPattern<SetErrorOp> {
 public:
  SetErrorOpLowering(TypeConverter &converter, MLIRContext *ctx,
                     Globals &globals)
      : OpConversionPattern(converter, ctx), globals_(globals) {}

  LogicalResult matchAndRewrite(
      SetErrorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get the error message (pointer to a null terminated string).
    auto err = Globals::OpaqueAddrOf(
        b, globals_.GetOrCreate(b, op.error(), "__assert_failed"));

    // Call runtime API to report the error.
    auto kernel_context = adaptor.ctx();
    rewriter.replaceOpWithNewOp<CallOp>(op, kSetError, TypeRange(),
                                        ValueRange({kernel_context, err}));

    return success();
  }

 private:
  Globals &globals_;
};

// -------------------------------------------------------------------------- //

class ConvertRuntimeToLLVMPass
    : public ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
 public:
  ConvertRuntimeToLLVMPass(CustomCallArgEncodingSet arg_encoding,
                           CustomCallAttrEncodingSet attr_encoding)
      : arg_encoding_(std::move(arg_encoding)),
        attr_encoding_(std::move(attr_encoding)) {}

  void runOnOperation() override;

  CustomCallArgEncodingSet arg_encoding_;
  CustomCallAttrEncodingSet attr_encoding_;
};

void ConvertRuntimeToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  // Add declarations for the runtime API functions.
  AddRuntimeApiDeclarations(module);

  RuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower all runtime operands to LLVM types.
  LLVMTypeConverter llvm_converter(ctx);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertKernelContextType);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertStatusType);

  // A helper class to create unique global constants.
  Globals globals(module);

  // Keep a cache of encoded values to encode each unique value just once.
  DenseMap<Value, CustomCallArgEncoding::Encoded> encoded_args;

  // Lower from the runtime operations to the runtime API function calls.
  patterns.add<SetOutputOpLowering, IsOkOpLowering>(llvm_converter, ctx);
  patterns.add<SetErrorOpLowering>(llvm_converter, ctx, globals);
  patterns.add<CustomCallOpLowering>(llvm_converter, ctx, globals,
                                     arg_encoding_, attr_encoding_,
                                     encoded_args);

  // Convert function signatures and call sites.
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Set up conversion target to rewrite all runtime operations.
  ConversionTarget target(*ctx);
  target.addIllegalDialect<RuntimeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ConstantOp, UnrealizedConversionCastOp, CallOp>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

CustomCallAttrEncodingSet DefaultAttrEncodings() {
  CustomCallAttrEncodingSet encodings;
  encodings.Add<StringAttrEncoding, ScalarAttrEncoding, ArrayAttrEncoding>();
  return encodings;
}

CustomCallArgEncodingSet DefaultArgEncodings() {
  CustomCallArgEncodingSet encodings;
  encodings.Add<ScalarArgEncoding, MemrefArgEncoding>();
  return encodings;
}

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass(
    CustomCallArgEncodingSet arg_encoding,
    CustomCallAttrEncodingSet attr_encoding) {
  return std::make_unique<ConvertRuntimeToLLVMPass>(std::move(arg_encoding),
                                                    std::move(attr_encoding));
}

}  // namespace jitrt
}  // namespace tfrt
