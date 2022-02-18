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

#ifndef TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_COMPILATION_UNITS_H_
#define TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_COMPILATION_UNITS_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// Resolve the properties of compilation units in the top level Module
// operation. Compilation units are functions in modules with `tfrt.compiled`
// attribute, which will be compiled at runtime by the BEF program, and passed
// as compilation unit attributes (serialized MLIR).
class BefCompilationUnits {
 public:
  explicit BefCompilationUnits(mlir::ModuleOp module) : module_(module) {}

  size_t SerializedSymbolId(mlir::SymbolRefAttr symbol);
  size_t SerializedSymbolSize(mlir::SymbolRefAttr symbol);
  size_t SerializedOperationSize(mlir::SymbolRefAttr symbol);

  ArrayRef<uint8_t> SerializedSymbolData(mlir::SymbolRefAttr symbol);
  ArrayRef<uint8_t> SerializedOperationData(mlir::SymbolRefAttr symbol);

  static bool IsCompiledModule(mlir::ModuleOp op);
  static bool IsInCompiledModule(mlir::Operation* op);

 private:
  struct Serialized {
    size_t id;              // sequential id of the serialized symbol
    size_t symbol_size;     // size of the serialized symbol name
    size_t operation_size;  // size of the serialized operation
    std::string data;       // symbol_ref + operation
  };

  const Serialized& Serialize(mlir::SymbolRefAttr symbol);

  mlir::ModuleOp module_;
  llvm::DenseMap<mlir::Attribute, Serialized> serialized_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_COMPILATION_UNITS_H_
