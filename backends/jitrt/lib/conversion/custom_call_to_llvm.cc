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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace tfrt {
namespace jitrt {

using llvm::StringRef;

using mlir::Attribute;
using mlir::failure;
using mlir::FailureOr;
using mlir::ImplicitLocOpBuilder;
using mlir::OpBuilder;
using mlir::succeeded;
using mlir::Type;
using mlir::Value;

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

}  // namespace jitrt
}  // namespace tfrt
