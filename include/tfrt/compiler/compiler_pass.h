/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// Compiler Pass
//
// This file declares CompilerPass interface.
#ifndef TFRT_COMPILER_COMPILER_PASS_H_
#define TFRT_COMPILER_COMPILER_PASS_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
// A simple compiler pass interface. This is meant to provide temporary
// interface before pluggable compiler pass has been fully designed.
// TODO(bramandia): Replace this with pluggable compiler passes once it is
// ready.
class CompilerPass {
 public:
  struct CompilationOutput {
    mlir::OwningModuleRef module;
    llvm::SmallVector<std::string, 4> output_devices;
  };

  virtual ~CompilerPass() {}

  virtual mlir::OwningModuleRef ParseMlirProgram(
      string_view program, mlir::MLIRContext* context) const = 0;
  virtual llvm::Expected<CompilationOutput> Compile(
      mlir::ModuleOp module, mlir::MLIRContext* context) const = 0;
};

void RegisterCompilerPass(const std::string& name, CompilerPass*);
const CompilerPass* GetCompilerPass(const std::string& name);
}  // namespace tfrt

#endif  // TFRT_COMPILER_COMPILER_PASS_H_
