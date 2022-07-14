/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tfrt/jitrt/conversion/custom_call_to_llvm.h"

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/jitrt/custom_call.h"

namespace tfrt {
namespace jitrt {

using llvm::StringRef;

using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::ComplexType;
using mlir::DenseIntOrFPElementsAttr;
using mlir::failure;
using mlir::FailureOr;
using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefDescriptor;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::succeeded;
using mlir::success;
using mlir::Type;
using mlir::TypeID;
using mlir::Value;
using mlir::ValueRange;

using mlir::arith::ConstantOp;

namespace LLVM = mlir::LLVM;

constexpr char CustomCallAttrEncoding::kAttrName[];
constexpr char CustomCallAttrEncoding::kAttrValue[];

// -------------------------------------------------------------------------- //
// Custom call arguments encoding.
// -------------------------------------------------------------------------- //

using EncodedArg = CustomCallArgEncodingSet::Encoded;

FailureOr<EncodedArg> CustomCallArgEncodingSet::Encode(Globals &g,
                                                       ImplicitLocOpBuilder &b,
                                                       Value value,
                                                       Value converted) const {
  for (auto &encoding : encodings_)
    if (succeeded(encoding->Match(value, converted)))
      return encoding->Encode(g, b, value, converted);
  return failure();
}

// -------------------------------------------------------------------------- //
// Custom call attributes encoding.
// -------------------------------------------------------------------------- //

using EncodedAttr = CustomCallAttrEncodingSet::Encoded;

FailureOr<EncodedAttr> CustomCallAttrEncodingSet::Encode(
    Globals &g, ImplicitLocOpBuilder &b, StringRef name, Attribute attr) const {
  for (auto &encoding : encodings_)
    if (succeeded(encoding->Match(name, attr)))
      return encoding->Encode(g, b, name, attr);
  return failure();
}

// -------------------------------------------------------------------------- //
// A set of helper functions for packing primitive types.
// -------------------------------------------------------------------------- //

// Packs TypeID as `i64` constant value and casts it to the `!llvm.ptr<i8>`,
// because type id internally is implemented as an opaque pointer.
Value PackTypeId(Globals &g, ImplicitLocOpBuilder &b, TypeID type_id) {
  auto i64 = reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer());
  auto cst = b.create<ConstantOp>(b.getI64IntegerAttr(i64));
  auto ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  return b.create<LLVM::IntToPtrOp>(ptr, cst);
}

// Packs string as a module global constants. Returns `!llvm.ptr<EncodedStr>`.
// We always pass string with the size to the runtime intrinsics, because
// computing the length of null-terminated string can be expensive, and we need
// it to construct llvm::StringRef at run time.
Value PackString(Globals &g, ImplicitLocOpBuilder &b, StringRef strref,
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

Value PackScalarAttribute(Globals &g, ImplicitLocOpBuilder &b, Attribute value,
                          StringRef symbol_base) {
  auto global = g.GetOrCreate(b, value, symbol_base);
  return Globals::AddrOf(b, global);
}

// Packs array attribute as a global constant. Returns `!llvm.ptr<EncodedArr>`.
Value PackDenseElementsAttribute(Globals &g, ImplicitLocOpBuilder &b,
                                 Attribute value, StringRef symbol_base) {
  MLIRContext *ctx = b.getContext();

  DenseIntOrFPElementsAttr dense = value.cast<DenseIntOrFPElementsAttr>();
  int64_t size = dense.getNumElements();

  // Encoded array type:
  // !llvm.struct<(i64, !llvm.ptr<array<element_type x size>>)>.
  Type element_type = dense.getElementType();
  Type arr_type = LLVM::LLVMArrayType::get(element_type, size);
  Type arr_ptr_type = LLVM::LLVMPointerType::get(arr_type);
  Type type =
      LLVM::LLVMStructType::getLiteral(ctx, {b.getI64Type(), arr_ptr_type});

  // Global constant initializer for the encoded array structure
  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    // Array size and the pointer to data.
    Value num_elements = ib.create<ConstantOp>(b.getI64IntegerAttr(size));
    Value data_ptr =
        Globals::AddrOf(ib, g.GetOrCreate(b, dense, arr_type, symbol_base, {}));

    // Store size and pointer into the struct.
    Value encoded = ib.create<LLVM::UndefOp>(type);
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, num_elements,
                                             ib.getI64ArrayAttr(0));
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, data_ptr,
                                             ib.getI64ArrayAttr(1));

    ib.create<LLVM::ReturnOp>(encoded);
  };

  auto global = g.GetOrCreate(b, value, type, symbol_base, init);
  return Globals::AddrOf(b, global);
}

// Create a global for the data array in an EncodedArray.
// Returns `!llvm.ptr<array<element_type x size>>
static Value CreateGlobalFromArray(Globals &g, ImplicitLocOpBuilder &b,
                                   ArrayAttr array, StringRef symbol_base) {
  Type element_type = array[0].getType();
  Type arr_type = LLVM::LLVMArrayType::get(element_type, array.size());

  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    Value data = ib.create<LLVM::UndefOp>(arr_type);
    for (int i = 0; i < array.size(); i++) {
      Value value = ib.create<ConstantOp>(array[i]);
      data = ib.create<LLVM::InsertValueOp>(arr_type, data, value,
                                            ib.getI64ArrayAttr(i));
    }
    ib.create<LLVM::ReturnOp>(data);
  };

  auto global = g.GetOrCreate(b, array, arr_type, symbol_base, init);
  return Globals::AddrOf(b, global);
}

Value PackArrayAttribute(Globals &g, ImplicitLocOpBuilder &b, Attribute value,
                         StringRef symbol_base) {
  MLIRContext *ctx = b.getContext();

  ArrayAttr array = value.cast<ArrayAttr>();
  int64_t size = array.size();

  // Encoded array type:
  // !llvm.struct<(i64, !llvm.ptr<array<element_type x size>)>>.
  Type element_type = array[0].getType();
  Type arr_type = LLVM::LLVMArrayType::get(element_type, size);
  Type arr_ptr_type = LLVM::LLVMPointerType::get(arr_type);
  Type type =
      LLVM::LLVMStructType::getLiteral(ctx, {b.getI64Type(), arr_ptr_type});

  // Global constant initializer for the encoded array structure
  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    // Array size and the pointer to data.
    Value num_elements = ib.create<ConstantOp>(b.getI64IntegerAttr(size));
    Value data = CreateGlobalFromArray(g, b, array, symbol_base);

    // Store size and values into the struct.
    Value encoded = ib.create<LLVM::UndefOp>(type);
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, num_elements,
                                             ib.getI64ArrayAttr(0));
    encoded = ib.create<LLVM::InsertValueOp>(type, encoded, data,
                                             ib.getI64ArrayAttr(1));

    ib.create<LLVM::ReturnOp>(encoded);
  };

  auto global = g.GetOrCreate(b, value, type, symbol_base, init);
  return Globals::AddrOf(b, global);
}

// Packs value on the stack. Returns `!llvm.ptr<ValueType>`.
Value PackValue(ImplicitLocOpBuilder &b, Value value) {
  Type ptr = LLVM::LLVMPointerType::get(value.getType());
  Value one = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(ptr, one, 0);
  b.create<LLVM::StoreOp>(value, mem);

  return mem;
}

// -------------------------------------------------------------------------- //
// A helper class to create global constants in the module.
// -------------------------------------------------------------------------- //

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

LLVM::GlobalOp Globals::GetOrCreate(ImplicitLocOpBuilder &b, Attribute attr,
                                    Type type, StringRef symbol_base,
                                    GlobalInitializer initialize) {
  Key key(attr, type, symbol_base);

  // Check if global value already exists ...
  if (auto global = Find(key)) return global;

  // ... otherwise create a new one.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module_.getBody());

  // If the initialize function is not provided, create constant directly.
  if (!initialize) {
    return b.create<LLVM::GlobalOp>(type, /*isConstant=*/true,
                                    LLVM::Linkage::Internal,
                                    UniqueSymName(symbol_base), attr);
  }

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
// -------------------------------------------------------------------------- //

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

static TypeID ArrayRuntimeTypeId(Type elem_type) {
  if (elem_type.isInteger(32)) return TypeID::get<Tagged<ArrayRef<int32_t>>>();
  if (elem_type.isInteger(64)) return TypeID::get<Tagged<ArrayRef<int64_t>>>();
  if (elem_type.isF32()) return TypeID::get<Tagged<ArrayRef<float>>>();
  if (elem_type.isF64()) return TypeID::get<Tagged<ArrayRef<double>>>();

  assert(false && "unsupported type id");
  return TypeID::getFromOpaquePointer(reinterpret_cast<void *>(0xDEADBEEF));
}

// -------------------------------------------------------------------------- //
// Custom call attributes encoding.
// -------------------------------------------------------------------------- //

LogicalResult StringAttrEncoding::Match(StringRef name, Attribute attr) const {
  return success(attr.isa<StringAttr>());
}

FailureOr<EncodedAttr> StringAttrEncoding::Encode(Globals &g,
                                                  ImplicitLocOpBuilder &b,
                                                  StringRef name,
                                                  Attribute attr) const {
  auto str = attr.cast<StringAttr>();

  Encoded encoded;
  encoded.name = PackString(g, b, name, kAttrName);
  encoded.type_id = PackTypeId(g, b, TypeID::get<Tagged<llvm::StringRef>>());
  encoded.value = PackString(g, b, str, kAttrValue);
  return encoded;
}

// -------------------------------------------------------------------------- //

LogicalResult ScalarAttrEncoding::Match(StringRef name, Attribute attr) const {
  return success(IsSupportedScalarType(attr.getType()));
}

FailureOr<EncodedAttr> ScalarAttrEncoding::Encode(Globals &g,
                                                  ImplicitLocOpBuilder &b,
                                                  StringRef name,
                                                  Attribute attr) const {
  Type type = attr.getType();

  Encoded encoded;
  encoded.name = PackString(g, b, name, kAttrName);
  encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
  encoded.value = PackScalarAttribute(g, b, attr, kAttrValue);

  return encoded;
}

// -------------------------------------------------------------------------- //

LogicalResult DenseElementsAttrEncoding::Match(StringRef name,
                                               Attribute attr) const {
  if (auto dense = attr.dyn_cast<DenseIntOrFPElementsAttr>())
    return success(IsSupportedShapedType(dense.getType()));
  return failure();
}

FailureOr<EncodedAttr> DenseElementsAttrEncoding::Encode(
    Globals &g, ImplicitLocOpBuilder &b, StringRef name, Attribute attr) const {
  Type elem_type = attr.getType().cast<ShapedType>().getElementType();

  Encoded encoded;
  encoded.name = PackString(g, b, name, kAttrName);
  encoded.type_id = PackTypeId(g, b, ArrayRuntimeTypeId(elem_type));
  encoded.value = PackDenseElementsAttribute(g, b, attr, kAttrValue);

  return encoded;
}

// -------------------------------------------------------------------------- //

LogicalResult ArrayAttrEncoding::Match(StringRef name, Attribute attr) const {
  if (auto array = attr.dyn_cast<ArrayAttr>()) {
    if (array.empty()) return failure();
    return success(IsSupportedScalarType(array[0].getType()));
  }
  return failure();
}

FailureOr<EncodedAttr> ArrayAttrEncoding::Encode(Globals &g,
                                                 ImplicitLocOpBuilder &b,
                                                 StringRef name,
                                                 Attribute attr) const {
  ArrayAttr array = attr.dyn_cast<ArrayAttr>();
  Type elem_type = array[0].getType();
  // We only support array attributes with elements of same type.
  for (Attribute attr : array) {
    if (attr.getType() != elem_type) {
      return failure();
    }
  }

  Encoded encoded;
  encoded.name = PackString(g, b, name, kAttrName);
  encoded.type_id = PackTypeId(g, b, ArrayRuntimeTypeId(elem_type));
  encoded.value = PackArrayAttribute(g, b, attr, kAttrValue);

  return encoded;
}

// -------------------------------------------------------------------------- //
// Encoding for aggregate attributes.
// -------------------------------------------------------------------------- //

mlir::FailureOr<mlir::Value> EncodeAggregateAttr(
    Globals &g, ImplicitLocOpBuilder &b,
    const CustomCallAttrEncodingSet &encoding, TypeID type_id,
    StringRef type_name, ArrayRef<NamedAttribute> attrs) {
  using EncodedAttr = std::pair<StringRef, CustomCallAttrEncoding::Encoded>;

  // In addition to encoded attributes we encode the number of attributes.
  int64_t n_attrs = attrs.size();

  // We store encoded aggregate attribute as `!llvm.array<ptr<i8> x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  Type type = LLVM::LLVMArrayType::get(ptr, 1 + n_attrs * 3);

  // Prepare a global constant for storing encoded aggregate.
  LLVM::GlobalOp global = [&]() {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(g.module().getBody());
    std::string global_name = ("__rt_aggregate_" + type_name).str();
    return b.create<LLVM::GlobalOp>(type, /*isConstant=*/true,
                                    LLVM::Linkage::Internal,
                                    g.UniqueSymName(global_name), Attribute());
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
    // Try to encode the attribute as a set of pointers.
    auto encoded_attr = encoding.Encode(g, b, attr.getName(), attr.getValue());
    if (failed(encoded_attr)) return failure();
    encoded.emplace_back(attr.getName(), *encoded_attr);
  }

  // Prepare an array for encoding attributes.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    Value bcasted = b.createOrFold<LLVM::BitcastOp>(ptr, value);
    arr = b.create<LLVM::InsertValueOp>(type, arr, bcasted,
                                        b.getI64ArrayAttr(offset));
  };

  // Insert the number of encoded attributes.
  Attribute num_attrs = b.getI64IntegerAttr(n_attrs);
  insert_value(PackScalarAttribute(g, b, num_attrs, "__rt_aggregate_size"), 0);

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

  // Return a pointer to the encoded aggregate: `!llvm.ptr<ptr<i8>>` (void**).
  return gep;
}

// -------------------------------------------------------------------------- //
// Custom call arguments encodings.
// -------------------------------------------------------------------------- //

LogicalResult ScalarArgEncoding::Match(Value value, Value converted) const {
  return success(IsSupportedScalarType(value.getType()));
}

FailureOr<EncodedArg> ScalarArgEncoding::Encode(Globals &g,
                                                ImplicitLocOpBuilder &b,
                                                Value value,
                                                Value converted) const {
  Type type = converted.getType();

  Encoded encoded;
  encoded.type_id = PackTypeId(g, b, ScalarRuntimeTypeId(type));
  encoded.value = PackValue(b, converted);

  return encoded;
}

// -------------------------------------------------------------------------- //

LogicalResult MemrefArgEncoding::Match(Value value, Value converted) const {
  return success(value.getType().isa<MemRefType>());
}

FailureOr<EncodedArg> MemrefArgEncoding::Encode(Globals &g,
                                                ImplicitLocOpBuilder &b,
                                                Value value,
                                                Value converted) const {
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

Value MemrefArgEncoding::EncodeMemRef(ImplicitLocOpBuilder &b,
                                      MemRefType memref_ty,
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

// -------------------------------------------------------------------------- //
// Default encodings for arguments and attributes.
// -------------------------------------------------------------------------- //

CustomCallAttrEncodingSet DefaultAttrEncodings() {
  CustomCallAttrEncodingSet encodings;
  encodings.Add<StringAttrEncoding, ScalarAttrEncoding,
                DenseElementsAttrEncoding, ArrayAttrEncoding>();
  return encodings;
}

CustomCallArgEncodingSet DefaultArgEncodings() {
  CustomCallArgEncodingSet encodings;
  encodings.Add<ScalarArgEncoding, MemrefArgEncoding>();
  return encodings;
}

}  // namespace jitrt
}  // namespace tfrt
