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

#ifndef RT_PASSES
#define RT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertRuntimeToLLVMPass : Pass<"rt-to-llvm", "ModuleOp"> {
  let summary = "Convert RT dialect to LLVM dialect";
  let constructor = "::tfrt::cpu::jit::CreateConvertRuntimeToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif  // RT_PASSES