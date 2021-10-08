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

def ConvertToKernelFunction : Pass<"rt-to-kernel-function", "ModuleOp"> {
  let summary = "Converts entrypoint function to kernel function";

  let description = [{
    Converts function with !rt.kernel_context as the first argument to a
    kernel function, i.e.:
     - all results returned via the `rt.set_result` operation
     - failed asserts set the kernel error via the `rt.set_error` operation

    Example:

      ```mlir
      func @compute(%ctx: !rt.kernel_context, ...) -> memref<?xf32> {
        %0 = ... : i1
        assert %0, "Oops"
        %1 = ... : memref<?xf32>
        return %0
      }
      ```

    converted to:

      ```mlir
      func @compute(%ctx: !rt.kernel_context, ...) {
        %0 = ... : i1
        cond_br %0, ^ok, ^err
      ^ok:
        %1 = ... : memref<?xf32>
        rt.set_output %ctx, 0, %1 : memref<xf32>
        return
      ^err:
        rt.set_error %ctx, "Oops"
        return;
      }
      ```
  }];

  let constructor = "::tfrt::cpu::jit::CreateConvertToKernelFunction()";
  let dependentDialects = ["tfrt::cpu::jit::RuntimeDialect"];
}

#endif  // RT_PASSES
