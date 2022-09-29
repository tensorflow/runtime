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

//===- Mlir-Opt utility ---------------------------------------------------===//
//
// Load MLIR and apply required passes on it.

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/passes/passes.h"
#include "tfrt/init_tfrt_dialects.h"
#include "tfrt/support/error_util.h"
#include "tfrt/test_kernels/opdefs/test_kernels.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  tfrt::compiler::TFRTDialect, tfrt::gpu::GpuDialect,
                  tfrt::test::TestDialect>();
  tfrt::gpu::RegisterPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TFRT pass driver\n", registry));
}
