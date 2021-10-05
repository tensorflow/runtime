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

//===- rt_ops.td -------------------------------------------------------===//
//
// Operation definitions for RT dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_OPS
#else
#define RT_OPS

include "tfrt/cpu/jit/opdefs/rt_base.td"

//===----------------------------------------------------------------------===//
// Op definitions.
//===----------------------------------------------------------------------===//

class RT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<RuntimeDialect, mnemonic, traits> {
}

def SetOutput : RT_Op<"set_output"> {
  let summary = "set the kernel output to a result value";

  let description = [{
    This operation sets the kernel output at the given index to the result
    value. In kernel functions we do not return the results using the
    conventional return statement, but use the runtime context API to pass
    values back to the runtime.

    Example:

      ```mlir
      func @compute(%ctx: !rt.kernel_context) {
        %out0 = ... : memref<?xf32>
        %out1 = ... : memref<?x?xf32>
        rt.set_output %ctx, 0, %out0 : memref<?xf32>
        rt.set_output %ctx, 1, %out1 : memref<?x?xf32>
      }
      ```

    is an equivalent of a regular function:

      ```mlir
      func @compute() -> (memref<?xf32>, memref<?x?xf32) {
        %out0 = ... : memref<?xf32>
        %out1 = ... : memref<?x?xf32>
        return %out0, %out1 : memref<?xf32>, memref<?x?xf32>
      }
      ```
  }];

  let arguments = (ins
    KernelContextType:$ctx,
    Confined<I64Attr, [IntNonNegative]>:$index,
    AnyType:$value
  );

  let assemblyFormat = [{
    $ctx `,` $index `,` $value `:` type($value) attr-dict
  }];
}

#endif // RT_OPS
