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

#ifndef CODEGEN_PASSES
#define CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def AlignedAllocationsPass : Pass<"codegen-aligned-allocations", "mlir::FuncOp"> {
  let summary = "Add alignment attribute to all `alloc` operations.";
  let description = [{
    This pass adds an alignment attribute to all `alloc` operations which don't
    have such an attribute yet, or which have a smaller alignment than the one
    configured for this pass.
  }];
  let constructor = "::tfrt::jitrt::CreateAlignedAllocationsPass()";
  let options = [
   Option<"alignment_", "alignment", "int64_t", "64",
          "Byte alignment for allocated memrefs.">
  ];
}

def MathOptimizationPass : Pass<"codegen-math-optimization", "mlir::FuncOp"> {
  let summary = "Optimize operations from the `math` dialect.";
  let description = [{
    This pass performs algebraic simplification and polynomial approximation for
    ops from the Math dialect.
  }];
  let constructor = "::tfrt::jitrt::CreateMathOptimizationPass()";
  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "mlir::x86vector::X86VectorDialect"
  ];
  let options = [
   Option<"enable_avx2_", "enable-avx2", "bool", "false",
          "Enable math approximations that emit AVX2 intrinsics.">
  ];
}

def CostDrivenAsyncParallelFor : Pass<"cost-driven-async-parallel-for", "mlir::ModuleOp"> {
  let summary = "Convert scf.parallel operations to multiple async compute ops "
                "executed concurrently for non-overlapping iteration ranges "
                "using a cost model to drive the determination of "
                "parallelization granularity";
  let constructor = "::tfrt::jitrt::CreateCostDrivenAsyncParallelForPass()";

  let options = [
    Option<"async_dispatch_", "async-dispatch",
      "bool", /*default=*/"true",
      "Dispatch async compute tasks using recursive work splitting. If `false` "
      "async compute tasks will be launched using simple for loop in the "
      "caller thread.">,

    Option<"num_worker_threads_", "num-workers",
      "int32_t", /*default=*/"8",
      "The number of available workers to execute async operations.">,

    // Temporary for bringup, should be removed.
    Option<"legacy_behavior_", "legacy-behavior", "bool",
            /*default=*/"true",
           "Emulate the upstream AsyncParallelFor behavior by producing a "
           "fixed cost for all inputs.">,
  ];

  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect",
    "::mlir::async::AsyncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

#endif  // CODEGEN_PASSES
