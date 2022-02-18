// Copyright 2021 The TensorFlow Runtime Authors
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

// BefCompilationUnits class to support SymbolRefAttr.

#include "bef_compilation_units.h"

#include <string>
#include <utility>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace tfrt {

bool BefCompilationUnits::IsCompiledModule(mlir::ModuleOp op) {
  return !!op->getAttr("tfrt.compiled");
}

bool BefCompilationUnits::IsInCompiledModule(mlir::Operation* op) {
  mlir::ModuleOp parent_module = dyn_cast<mlir::ModuleOp>(op);
  if (!parent_module) parent_module = op->getParentOfType<mlir::ModuleOp>();

  while (parent_module) {
    if (IsCompiledModule(parent_module)) return true;
    parent_module = parent_module->getParentOfType<mlir::ModuleOp>();
  }

  return false;
}

size_t BefCompilationUnits::SerializedSymbolId(mlir::SymbolRefAttr symbol) {
  return Serialize(symbol).id;
}

size_t BefCompilationUnits::SerializedSymbolSize(mlir::SymbolRefAttr symbol) {
  return Serialize(symbol).symbol_size;
}

size_t BefCompilationUnits::SerializedOperationSize(
    mlir::SymbolRefAttr symbol) {
  return Serialize(symbol).operation_size;
}

ArrayRef<uint8_t> BefCompilationUnits::SerializedSymbolData(
    mlir::SymbolRefAttr symbol) {
  auto& serialized = Serialize(symbol);
  string_view str = serialized.data;
  return {reinterpret_cast<const uint8_t*>(str.data()), serialized.symbol_size};
}

ArrayRef<uint8_t> BefCompilationUnits::SerializedOperationData(
    mlir::SymbolRefAttr symbol) {
  auto& serialized = Serialize(symbol);
  string_view str = serialized.data;
  return {reinterpret_cast<const uint8_t*>(str.data()) + serialized.symbol_size,
          serialized.operation_size};
}

const BefCompilationUnits::Serialized& BefCompilationUnits::Serialize(
    mlir::SymbolRefAttr symbol) {
  auto* op = mlir::SymbolTable::lookupSymbolIn(module_.getOperation(), symbol);
  assert(IsInCompiledModule(op));

  // Check if the referenced symbol already serialized.
  auto it = serialized_.find(symbol);
  if (it != serialized_.end()) return it->getSecond();

  // Serialize and keep compiled module that defines the symbol.
  auto parent_module = op->getParentOfType<mlir::ModuleOp>();
  assert(IsCompiledModule(parent_module));

  std::string str;
  llvm::raw_string_ostream os(str);

  // Print symbol names.
  os << symbol.getRootReference().getValue();
  for (auto nested_ref : symbol.getNestedReferences())
    os << nested_ref.getValue();

  size_t symbol_size = str.size();

  // Use generic form to print the module and improve BEF portability. The
  // pretty print is less stable from a syntax point of view.
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm();
  parent_module.print(os, flags);

  size_t id = serialized_.size();
  size_t operation_size = str.size() - symbol_size;
  Serialized serialized{id, symbol_size, operation_size, std::move(str)};

  auto emplaced = serialized_.try_emplace(symbol, std::move(serialized));
  assert(emplaced.second && "emplace must be successful");
  return emplaced.first->getSecond();
}

}  // namespace tfrt
