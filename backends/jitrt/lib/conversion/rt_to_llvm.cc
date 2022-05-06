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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
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
#include "tfrt/jitrt/conversion/rt_passes.h"
#include "tfrt/jitrt/custom_call.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"

namespace tfrt {
namespace jitrt {
namespace {

using llvm::DenseMap;

using mlir::Attribute;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DenseIntOrFPElementsAttr;
using mlir::failure;
using mlir::FailureOr;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::IntegerType;
using mlir::LLVMTypeConverter;
using mlir::Location;
using mlir::LogicalResult;
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
};

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(ModuleOp module) {
  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  auto addDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name)) return;
    b.create<FuncOp>(name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addDecl(kGetResultStorage, RuntimeAPI::GetResultStorageFunctionType(ctx));
  addDecl(kSetError, RuntimeAPI::SetErrorFunctionType(ctx));
  addDecl(kCustomCall, RuntimeAPI::CustomCallFunctionType(ctx));
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

// Arguments to the custom call API intrinsic are encoded as an array of opaque
// pointers and at the runtime side available as `void**`. Runtime decodes
// opaque pointers to the C++ data structures (see jitrt/custom_call.h), and
// passes them to the registered callback. Argument encoding/decoding must be
// compatible, otherwise it's very easy to get a segfault because of an illegal
// memory access.
//
// Attributes are encoded into a separate opaque storage together with names, so
// the runtime side can decode the attributes it needs and check that all
// required attributes were passed to the custom call handler.
//
// Custom call attributes are encoded as module global constants, and at run
// time we only need to pass a pointer to the constant section.
//
// Custom call arguments are encoded as an array of pointers allocated on the
// stack. Each individual argument is also encoded on the stack, because
// arguments are run time values and we can't encode them in the constant
// section.

// -------------------------------------------------------------------------- //
// A helper class to create global constants at the top of the module.

namespace {
class Globals {
 public:
  explicit Globals(ModuleOp module) : module_(module) {}

  // Returns a unique symbol name for a given `symbol_base`.
  std::string UniqueSymName(StringRef symbol_base);

  // Creates a global null-terminated string constant.
  LLVM::GlobalOp GetOrCreate(ImplicitLocOpBuilder &b, StringRef strref,
                             StringRef symbol_base);

  // Creates a global constant value from the attribute. Attribute type must be
  // a valid type compatible with LLVM globals.
  LLVM::GlobalOp GetOrCreate(ImplicitLocOpBuilder &b, Attribute attr,
                             StringRef symbol_base);

  // Creates a global constant value of the given type from the attribute, using
  // user-provided global constant initialization.
  LLVM::GlobalOp GetOrCreate(
      ImplicitLocOpBuilder &b, Attribute attr, Type type, StringRef symbol_base,
      llvm::function_ref<void(ImplicitLocOpBuilder &, Attribute)> initialize);

  // Returns the address of the global value.
  static Value AddrOf(ImplicitLocOpBuilder &b, LLVM::GlobalOp global);

  // Return the address of the global value casted to `!llvm.ptr<i8>`.
  static Value OpaqueAddrOf(ImplicitLocOpBuilder &b, LLVM::GlobalOp global);

  ModuleOp module() { return module_; }

 private:
  // Globals key: {attribute, encoded-type, sym-name}. We can only have global
  // constants of one of the LLVM types, and there could be multiple ways to
  // encode an attribute as an LLVM type, e.g. strings can be stored as null
  // terminated array of bytes, or a pair of string size and and array of bytes.
  using Key = std::tuple<Attribute, Type, StringRef>;

  LLVM::GlobalOp Find(Key key);

  ModuleOp module_;
  DenseMap<Key, LLVM::GlobalOp> globals_;
};
}  // namespace

std::string Globals::UniqueSymName(StringRef symbol_base) {
  int cnt = 0;
  std::string str = symbol_base.str();

  mlir::SymbolTable sym_table(module_);
  while (sym_table.lookup(str))
    str = llvm::formatv("{0}_{1}", symbol_base, cnt++);

  return str;
}

LLVM::GlobalOp Globals::Find(Key key) {
  auto it = globals_.find(key);
  if (it != globals_.end()) return it->second;
  return nullptr;
}

LLVM::GlobalOp Globals::GetOrCreate(ImplicitLocOpBuilder &b, StringRef strref,
                                    StringRef symbol_base) {
  // Create an std::string to get a null terminated sequence of characters.
  std::string str = strref.str();

  // Create a string reference that captures the null terminator.
  StringRef ref(str.data(), str.size() + 1);
  auto arr = LLVM::LLVMArrayType::get(b.getI8Type(), ref.size());

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module_.getBody());

  return b.create<LLVM::GlobalOp>(
      arr, /*isConstant=*/true, LLVM::Linkage::Internal,
      UniqueSymName(symbol_base), b.getStringAttr(ref));
}

LLVM::GlobalOp Globals::GetOrCreate(ImplicitLocOpBuilder &b, Attribute attr,
                                    StringRef symbol_base) {
  Key key(attr, attr.getType(), symbol_base);

  // Check if global value already exists ...
  if (auto global = Find(key)) return global;

  // ... otherwise create a new one.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module_.getBody());

  auto global = b.create<LLVM::GlobalOp>(attr.getType(), /*isConstant=*/true,
                                         LLVM::Linkage::Internal,
                                         UniqueSymName(symbol_base), attr);
  auto emplaced = globals_.try_emplace(key, global);
  assert(emplaced.second && "must be a new global");

  return emplaced.first->second;
}

LLVM::GlobalOp Globals::GetOrCreate(
    ImplicitLocOpBuilder &b, Attribute attr, Type type, StringRef symbol_base,
    llvm::function_ref<void(ImplicitLocOpBuilder &, Attribute)> initialize) {
  Key key(attr, type, symbol_base);

  // Check if global value already exists ...
  if (auto global = Find(key)) return global;

  // ... otherwise create a new one.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module_.getBody());

  // Create an uninitialized global.
  auto global = b.create<LLVM::GlobalOp>(
      type, /*isConstant=*/true, LLVM::Linkage::Internal,
      UniqueSymName(symbol_base), Attribute());
  auto emplaced = globals_.try_emplace(key, global);
  assert(emplaced.second && "must be a new global");

  // Call user-provided global initializer.
  mlir::Region &region = global.getInitializerRegion();
  mlir::Block *block = b.createBlock(&region);

  b.setInsertionPointToStart(block);
  initialize(b, attr);

  return emplaced.first->second;
}

/*static*/ Value Globals::AddrOf(ImplicitLocOpBuilder &b,
                                 LLVM::GlobalOp global) {
  return b.create<LLVM::AddressOfOp>(
      LLVM::LLVMPointerType::get(global.getType()), global.getSymName());
}

/*static*/ Value Globals::OpaqueAddrOf(ImplicitLocOpBuilder &b,
                                       LLVM::GlobalOp global) {
  return b.create<LLVM::BitcastOp>(LLVM::LLVMPointerType::get(b.getI8Type()),
                                   AddrOf(b, global));
}

// -------------------------------------------------------------------------- //
// Helper functions for encoding attributes and values for custom calls.

static bool IsSupportedScalarType(Type type) {
  auto is_supported_width = [](unsigned width, ArrayRef<unsigned> supported) {
    return llvm::any_of(supported, [&](unsigned w) { return w == width; });
  };

  if (auto integer = type.dyn_cast<mlir::IntegerType>())
    return is_supported_width(integer.getWidth(), {32, 64});

  if (auto fp = type.dyn_cast<mlir::FloatType>())
    return is_supported_width(fp.getWidth(), {32, 64});

  return false;
}

static bool IsSupportedShapedType(ShapedType shape) {
  return shape.getRank() == 1 && IsSupportedScalarType(shape.getElementType());
  return false;
}

static TypeID ScalarRuntimeTypeId(Type type) {
  if (type.isUnsignedInteger(8)) return TypeID::get<uint8_t>();
  if (type.isUnsignedInteger(32)) return TypeID::get<uint32_t>();
  if (type.isUnsignedInteger(64)) return TypeID::get<uint64_t>();

  if (type.isInteger(32)) return TypeID::get<int32_t>();
  if (type.isInteger(64)) return TypeID::get<int64_t>();

  if (type.isF32()) return TypeID::get<float>();
  if (type.isF64()) return TypeID::get<double>();

  assert(false && "unsupported type id");
  return TypeID::getFromOpaquePointer(reinterpret_cast<void *>(0xDEADBEEF));
}

static TypeID ArrayRuntimeTypeId(Type shaped) {
  auto type = shaped.cast<ShapedType>().getElementType();
  assert(shaped.cast<ShapedType>().getRank() == 1 && "unsupported rank");

  if (type.isInteger(32)) return TypeID::get<ArrayRef<int32_t>>();
  if (type.isInteger(64)) return TypeID::get<ArrayRef<int64_t>>();
  if (type.isF32()) return TypeID::get<ArrayRef<float>>();
  if (type.isF64()) return TypeID::get<ArrayRef<double>>();

  assert(false && "unsupported type id");
  return TypeID::getFromOpaquePointer(reinterpret_cast<void *>(0xDEADBEEF));
}

// Packs scalar attribute as a global constant. Returns `!llvm.ptr<AttrType>`.
static Value PackScalarAttribute(Globals &g, ImplicitLocOpBuilder &b,
                                 Attribute value, StringRef symbol_base) {
  auto global = g.GetOrCreate(b, value, symbol_base);
  return Globals::AddrOf(b, global);
}

// Packs TypeID as a global constant. Returns `!llvm.ptr<i64>`.
static Value PackTypeId(Globals &g, ImplicitLocOpBuilder &b, TypeID type_id) {
  auto i64 = reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer());
  auto global = g.GetOrCreate(b, b.getI64IntegerAttr(i64), "__rt_type_id");
  return Globals::AddrOf(b, global);
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

// Attributes encoding packs attribute name, data type and a value into the
// module global constant, and returns values pointing to the encoded data.
struct CustomCallAttrEncoding {
  static constexpr char kAttrName[] = "__rt_attr_name";
  static constexpr char kAttrValue[] = "__rt_attr_value";

  struct Encoded {
    Value name;     // !llvm.ptr<i8>
    Value type_id;  // !llvm.ptr<i64>
    Value value;    // !llvm.ptr<EncodedAttrType>
  };

  virtual ~CustomCallAttrEncoding() = default;

  virtual FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b,
                                    StringRef name, Attribute value) const = 0;
};

struct StringAttrEncoding : public CustomCallAttrEncoding {
  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute value) const override {
    auto str = value.cast<StringAttr>();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, TypeID::get<llvm::StringRef>());
    encoded.value = PackString(g, b, str, kAttrValue);
    return encoded;
  }
};

constexpr char CustomCallAttrEncoding::kAttrName[];
constexpr char CustomCallAttrEncoding::kAttrValue[];

struct ScalarAttrEncoding : public CustomCallAttrEncoding {
  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute value) const override {
    Type type = value.getType();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
    encoded.value = PackScalarAttribute(g, b, value, kAttrValue);

    return encoded;
  }
};

struct ArrayAttrEncoding : public CustomCallAttrEncoding {
  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, StringRef name,
                            Attribute value) const override {
    ShapedType type = value.getType().cast<ShapedType>();

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, ArrayRuntimeTypeId(type));
    encoded.value = PackArrayAttribute(g, b, value, kAttrValue);

    return encoded;
  }
};

// -------------------------------------------------------------------------- //

// Encodes argument into stack allocated storage according to the ABI. If
// argument is a constant, then it can be packed as a global constant.
class CustomCallArgEncoding {
 public:
  struct Encoded {
    Value type_id;  // !llvm.ptr<i64>
    Value value;    // !llvm.ptr<ArgType>
  };

  virtual ~CustomCallArgEncoding() = default;

  virtual FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b,
                                    Value value, Value converted) const = 0;
};

// Encodes scalar operands.
class ScalarArgEncoding : public CustomCallArgEncoding {
 public:
  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, Value value,
                            Value converted) const override {
    Type type = converted.getType();

    Encoded encoded;
    encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
    encoded.value = PackValue(b, converted);

    return encoded;
  }
};

// Encodes MemRef operands according to the MemrefView ABI.
class MemrefArgEncoding : public CustomCallArgEncoding {
 public:
  FailureOr<Encoded> Encode(Globals &g, ImplicitLocOpBuilder &b, Value value,
                            Value converted) const override {
    auto memref_type = value.getType().cast<MemRefType>();

    Encoded encoded;
    encoded.type_id = PackTypeId(g, b, TypeID::get<MemrefView>());
    encoded.value = PackValue(b, EncodeMemRef(b, memref_type, converted));

    return encoded;
  }

 private:
  // Encodes memref as the LLVM structure value: type id (i64), rank (i64) and a
  // pointer to the strided memref descriptor (ptr<i8>).
  Value EncodeMemRef(ImplicitLocOpBuilder &b, MemRefType memref_ty,
                     Value descriptor) const {
    MLIRContext *ctx = b.getContext();

    // Encoded memref type: !llvm.struct<(i64, i64, ptr<i8>)>.
    Type i64 = b.getI64Type();
    Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
    Type type = LLVM::LLVMStructType::getLiteral(ctx, {i64, i64, ptr});

    TypeID runtime_type_id = ScalarRuntimeTypeId(memref_ty.getElementType());

    // Create values for filling encoded memref struct.
    Value type_id = b.create<ConstantOp>(
        b.getI64IntegerAttr(reinterpret_cast<std::uintptr_t>(
            runtime_type_id.getAsOpaquePointer())));
    Value rank = b.create<ConstantOp>(b.getI64IntegerAttr(memref_ty.getRank()));
    Value desc = b.create<LLVM::BitcastOp>(ptr, PackValue(b, descriptor));

    auto offset = [&](int64_t i) { return b.getI64ArrayAttr(i); };

    // Create undef value for encoded memref.
    Value memref = b.create<LLVM::UndefOp>(type);
    memref = b.create<LLVM::InsertValueOp>(type, memref, type_id, offset(0));
    memref = b.create<LLVM::InsertValueOp>(type, memref, rank, offset(1));
    memref = b.create<LLVM::InsertValueOp>(type, memref, desc, offset(2));

    return memref;
  }
};

// ------------------------------------------------------------------------- -//

// TODO(ezhulenev): Support dynamic encoding registration for arguments.
static FailureOr<CustomCallArgEncoding::Encoded> EncodeArgument(
    Globals &g, ImplicitLocOpBuilder &b,
    std::tuple<Value, Value> value_and_converted) {
  Value value = std::get<0>(value_and_converted);
  Value converted = std::get<1>(value_and_converted);

  // Scalar arguments encoding.
  if (IsSupportedScalarType(value.getType()))
    return ScalarArgEncoding().Encode(g, b, value, converted);

  // Memref arguments encoding.
  if (value.getType().isa<MemRefType>())
    return MemrefArgEncoding().Encode(g, b, value, converted);

  return failure();
}

// TODO(ezhulenev): Support dynamic encoding registration for attributes.
static FailureOr<CustomCallAttrEncoding::Encoded> EncodeAttribute(
    Globals &g, ImplicitLocOpBuilder &b, NamedAttribute attr) {
  StringRef name = attr.getName();
  Attribute value = attr.getValue();

  // String attributes encoding.
  if (value.isa<StringAttr>())
    return StringAttrEncoding().Encode(g, b, name, value);

  // Scalar attributes encoding.
  if (IsSupportedScalarType(value.getType()))
    return ScalarAttrEncoding().Encode(g, b, name, value);

  // Dense attributes encoding.
  if (auto dense = value.dyn_cast<DenseIntOrFPElementsAttr>())
    if (IsSupportedShapedType(dense.getType()))
      return ArrayAttrEncoding().Encode(g, b, name, value);

  // TODO(ezhulenev): Support `ArrayAttr` with scalar elements.

  return failure();
}

static FailureOr<Value> EncodeArguments(
    Globals &g, DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args,
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

    auto encoded_arg = EncodeArgument(g, b, tuple);
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
static FailureOr<Value> EncodeAttributes(Globals &g, ImplicitLocOpBuilder &b,
                                         ArrayRef<NamedAttribute> attrs) {
  using EncodedAttr = std::pair<StringRef, CustomCallAttrEncoding::Encoded>;

  // Callee passed explicitly as a custom call argument.
  auto skip = [](NamedAttribute attr) { return attr.getName() == "callee"; };

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

  // Create a global constant initializer block.
  mlir::Region &region = global.getInitializerRegion();
  mlir::Block *block = b.createBlock(&region);

  // Build attributes encoding inside the initializer block.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(block);

  llvm::SmallVector<EncodedAttr> encoded;
  for (auto &attr : attrs) {
    if (skip(attr)) continue;

    // Try to encode the attribute as a set of pointers.
    auto encoded_attr = EncodeAttribute(g, b, attr);
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
      DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args)
      : OpConversionPattern(converter, ctx),
        globals_(globals),
        encoded_args_(encoded_args) {}

  LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get the custom call target (pointer to a null terminated string).
    auto callee = Globals::OpaqueAddrOf(
        b, globals_.GetOrCreate(b, op.callee(), "__rt_custom_call_callee"));

    // Encode operation arguments as a runtime API arguments.
    auto args = EncodeArguments(globals_, encoded_args_, b, op->getOperands(),
                                adaptor.getOperands());
    if (failed(args)) return op.emitOpError() << "failed to encode arguments";

    // Encode operation attributes as a runtime API argument.
    auto attrs = EncodeAttributes(globals_, b, op->getAttrs());
    if (failed(attrs)) return op.emitOpError() << "failed to encode attributes";

    // Call runtime API to call the custom call target.
    auto i1 = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, kCustomCall, TypeRange(i1),
        ValueRange({adaptor.ctx(), callee, *args, *attrs}));

    return success();
  }

 private:
  Globals &globals_;
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
  void runOnOperation() override;
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

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass() {
  return std::make_unique<ConvertRuntimeToLLVMPass>();
}

}  // namespace jitrt
}  // namespace tfrt
