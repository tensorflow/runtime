/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TFRT_BACKENDS_JITRT_CONVERSION_CUSTOM_CALL_TO_LLVM_H_
#define TFRT_BACKENDS_JITRT_CONVERSION_CUSTOM_CALL_TO_LLVM_H_

#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/jitrt/custom_call.h"

namespace tfrt {
namespace jitrt {

// -------------------------------------------------------------------------- //
// Helper classes to build a JitRt custom calls lowering to the LLVM dialect.
// -------------------------------------------------------------------------- //
//
// Arguments to the custom call API intrinsic are encoded as an array of opaque
// pointers and at the runtime side available as `void**`. Runtime decodes
// opaque pointers to the C++ data structures (see jitrt/custom_call.h), and
// passes them to the registered callback. Argument encoding/decoding must be
// compatible, otherwise it's very easy to get a segfault because of an illegal
// memory access.
//
// Attributes are encoded into a separate opaque storage together with names,
// so the runtime side can decode the attributes it needs and check that all
// required attributes were passed to the custom call handler.
//
// Custom call attributes are encoded as module global constants, and at run
// time we only need to pass a pointer to the constant section.
//
// Custom call arguments are encoded as an array of pointers allocated on the
// stack. Each individual argument is also encoded on the stack, because
// arguments are run time values and we can't encode them in the constant
// section.

// Forward declare class declared below.
class Globals;

// -------------------------------------------------------------------------- //
// Custom call arguments encoding.
// -------------------------------------------------------------------------- //

// Encodes argument into stack allocated storage according to the ABI. If
// argument is a constant, then it can be packed as a global constant.
class CustomCallArgEncoding {
 public:
  struct Encoded {
    mlir::Value type_id;  // !llvm.ptr<i64>
    mlir::Value value;    // !llvm.ptr<ArgType>
  };

  virtual ~CustomCallArgEncoding() = default;

  virtual mlir::LogicalResult Match(mlir::Value value,
                                    mlir::Value conterted) const = 0;

  virtual mlir::FailureOr<Encoded> Encode(Globals &g,
                                          mlir::ImplicitLocOpBuilder &b,
                                          mlir::Value value,
                                          mlir::Value converted) const = 0;
};

// A set of registered custom call arguments encodings.
class CustomCallArgEncodingSet {
 public:
  using Encoded = CustomCallArgEncoding::Encoded;

  // Finds matching argument encoding and tries to encode the values. Returns
  // failure if didn't match values to any of the argument encodings.
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::Value value,
                                  mlir::Value converted) const;

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallArgEncodingSet &Add() {
    (void)std::initializer_list<int>{
        0, (encodings_.emplace_back(std::make_shared<Ts>()), 0)...};
    return *this;
  }

 private:
  std::vector<std::shared_ptr<CustomCallArgEncoding>> encodings_;
};

// -------------------------------------------------------------------------- //
// Custom call attributes encoding.
// -------------------------------------------------------------------------- //

// Attributes encoding packs attribute name, data type and a value into the
// module global constant, and returns values pointing to the encoded data.
struct CustomCallAttrEncoding {
  static constexpr char kAttrName[] = "__rt_attr_name";
  static constexpr char kAttrValue[] = "__rt_attr_value";

  struct Encoded {
    mlir::Value name;     // !llvm.ptr<i8>
    mlir::Value type_id;  // !llvm.ptr<i64>
    mlir::Value value;    // !llvm.ptr<EncodedAttrType>
  };

  virtual ~CustomCallAttrEncoding() = default;

  virtual mlir::LogicalResult Match(llvm::StringRef name,
                                    mlir::Attribute attr) const = 0;

  virtual mlir::FailureOr<Encoded> Encode(Globals &g,
                                          mlir::ImplicitLocOpBuilder &b,
                                          llvm::StringRef name,
                                          mlir::Attribute attr) const = 0;
};

// A set of registered custom call attributes encodings.
class CustomCallAttrEncodingSet {
 public:
  using Encoded = CustomCallAttrEncoding::Encoded;

  // Finds matching attribute encoding and tries to encode the attribute.
  // Returns failure if didn't match attribute to any of the encodings.
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  llvm::StringRef name,
                                  mlir::Attribute attr) const;

  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  CustomCallAttrEncodingSet &Add() {
    (void)std::initializer_list<int>{
        0, (encodings_.emplace_back(std::make_shared<Ts>()), 0)...};
    return *this;
  }

 private:
  std::vector<std::shared_ptr<CustomCallAttrEncoding>> encodings_;
};

// -------------------------------------------------------------------------- //
// A set of helper functions for packing attributes and values.
// -------------------------------------------------------------------------- //

// Packs TypeID as `i64` constant value and casts it to the `!llvm.ptr<i8>`,
// because type id internally is implemented as an opaque pointer.
mlir::Value PackTypeId(Globals &g, mlir::ImplicitLocOpBuilder &b,
                       mlir::TypeID type_id);

// Packs string as a module global constants. Returns `!llvm.ptr<EncodedStr>`.
// We always pass string with the size to the runtime intrinsics, because
// computing the length of null-terminated string can be expensive, and we need
// it to construct llvm::StringRef at run time.
mlir::Value PackString(Globals &g, mlir::ImplicitLocOpBuilder &b,
                       llvm::StringRef strref, llvm::StringRef symbol_base);

// Packs scalar attribute as a global constant. Returns `!llvm.ptr<AttrType>`.
mlir::Value PackScalarAttribute(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                mlir::Attribute value,
                                mlir::StringRef symbol_base);

// Packs array attribute as a global constant. Returns `!llvm.ptr<EncodedArr>`.
mlir::Value PackArrayAttribute(Globals &g, mlir::ImplicitLocOpBuilder &b,
                               mlir::Attribute value,
                               llvm::StringRef symbol_base);

// Packs value on the stack. Returns `!llvm.ptr<ValueType>`.
mlir::Value PackValue(mlir::ImplicitLocOpBuilder &b, mlir::Value value);

// -------------------------------------------------------------------------- //
// A helper class to create global constants in the module.
// -------------------------------------------------------------------------- //

class Globals {
 public:
  // Global value initializer that build the initialization region.
  using GlobalInitializer =
      std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Attribute)>;

  explicit Globals(mlir::ModuleOp module) : module_(module) {}

  // Returns a unique symbol name for a given `symbol_base`.
  std::string UniqueSymName(llvm::StringRef symbol_base);

  // Creates a global null-terminated string constant.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   llvm::StringRef strref,
                                   llvm::StringRef symbol_base);

  // Creates a global constant value from the attribute. Attribute type must be
  // a valid type compatible with LLVM globals.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   mlir::Attribute attr,
                                   llvm::StringRef symbol_base);

  // Creates a global constant value of the given type from the attribute, using
  // user-provided global constant initialization.
  mlir::LLVM::GlobalOp GetOrCreate(mlir::ImplicitLocOpBuilder &b,
                                   mlir::Attribute attr, mlir::Type type,
                                   llvm::StringRef symbol_base,
                                   GlobalInitializer initialize);

  // Returns the address of the global value.
  static mlir::Value AddrOf(mlir::ImplicitLocOpBuilder &b,
                            mlir::LLVM::GlobalOp global);

  // Return the address of the global value casted to `!llvm.ptr<i8>`.
  static mlir::Value OpaqueAddrOf(mlir::ImplicitLocOpBuilder &b,
                                  mlir::LLVM::GlobalOp global);

  mlir::ModuleOp module() { return module_; }

 private:
  // Globals key: {attribute, encoded-type, sym-name}. We can only have global
  // constants of one of the LLVM types, and there could be multiple ways to
  // encode an attribute as an LLVM type, e.g. strings can be stored as null
  // terminated array of bytes, or a pair of string size and and array of bytes.
  using Key = std::tuple<mlir::Attribute, mlir::Type, llvm::StringRef>;

  mlir::LLVM::GlobalOp Find(Key key);

  mlir::ModuleOp module_;
  llvm::DenseMap<Key, mlir::LLVM::GlobalOp> globals_;
};

// -------------------------------------------------------------------------- //
// Custom call attributes encoding.
// -------------------------------------------------------------------------- //

struct StringAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(llvm::StringRef, mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::StringRef, mlir::Attribute) const final;
};

struct ScalarAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(llvm::StringRef, mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::StringRef, mlir::Attribute) const final;
};

struct ArrayAttrEncoding : public CustomCallAttrEncoding {
  mlir::LogicalResult Match(llvm::StringRef, mlir::Attribute) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::StringRef, mlir::Attribute) const final;
};

// Custom call attribute encoding that encodes enums using their underlying
// scalar type. Type id is based on the actual enum type passed to the runtime.
//
// TODO(ezhulenev): Add an option to define enum conversion function when
// the enum type at compile time doesn't match the enum type at run time.
template <typename AttrType, typename EnumType>
struct EnumAttrEncoding : public CustomCallAttrEncoding {
  static_assert(std::is_enum<EnumType>::value, "type must be an enum class");

  using T = std::underlying_type_t<EnumType>;

  mlir::LogicalResult Match(llvm::StringRef, mlir::Attribute attr) const final {
    return mlir::success(attr.isa<AttrType>());
  }

  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::StringRef name,
                                  mlir::Attribute attr) const final {
    // Convert enum underlying integral value to an attribute.
    T underlying_value = static_cast<T>(attr.cast<AttrType>().getValue());
    mlir::Attribute underlying_attr = AsAttr(b, underlying_value);

    Encoded encoded;
    encoded.name = PackString(g, b, name, kAttrName);
    encoded.type_id = PackTypeId(g, b, mlir::TypeID::get<Tagged<EnumType>>());
    encoded.value = PackScalarAttribute(g, b, underlying_attr, kAttrValue);

    return encoded;
  }

  static mlir::Attribute AsAttr(mlir::ImplicitLocOpBuilder &b, uint32_t value) {
    return b.getI32IntegerAttr(value);
  }
};

// -------------------------------------------------------------------------- //
// Custom call arguments encoding.
// -------------------------------------------------------------------------- //

// Encodes scalar operands.
class ScalarArgEncoding : public CustomCallArgEncoding {
 public:
  mlir::LogicalResult Match(mlir::Value, mlir::Value) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::Value, mlir::Value) const final;
};

// Encodes MemRef operands according to the (Strided)MemrefView ABI.
class MemrefArgEncoding : public CustomCallArgEncoding {
 public:
  mlir::LogicalResult Match(mlir::Value, mlir::Value) const final;
  mlir::FailureOr<Encoded> Encode(Globals &g, mlir::ImplicitLocOpBuilder &b,
                                  mlir::Value, mlir::Value) const final;

 private:
  // Encodes memref as LLVM struct value:
  //
  //   { i8: dtype, i8: rank, ptr<i8>: data,
  //     array<2*rank x i64>: sizes_and_strides }
  //
  // This is a type erased version of the MLIR memref descriptor without base
  // pointer. We pack sizes and strides as a single array member, so that on
  // the runtime side we can read it back using C flexible array member.
  mlir::Value EncodeMemRef(mlir::ImplicitLocOpBuilder &b,
                           mlir::MemRefType memref_ty,
                           mlir::Value descriptor) const;
};

// -------------------------------------------------------------------------- //
// Default encodings for arguments and attributes.
// -------------------------------------------------------------------------- //

CustomCallArgEncodingSet DefaultArgEncodings();
CustomCallAttrEncodingSet DefaultAttrEncodings();

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_CONVERSION_CUSTOM_CALL_TO_LLVM_H_
