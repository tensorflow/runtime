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
    "::mlir::arith::ArithDialect",
    "::mlir::async::AsyncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

#endif  // CODEGEN_PASSES
