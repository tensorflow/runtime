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

include "tfrt/jitrt/opdefs/rt_base.td"

//===----------------------------------------------------------------------===//
// Op definitions.
//===----------------------------------------------------------------------===//

class RT_Op<string mnemonic, list<Trait> traits = []> :
      Op<RuntimeDialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// SetOutputOp
//===----------------------------------------------------------------------===//

def SetOutputOp : RT_Op<"set_output"> {
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

//===----------------------------------------------------------------------===//
// SetErrorOp
//===----------------------------------------------------------------------===//

def SetErrorOp : RT_Op<"set_error"> {
  let summary = "set all kernel outputs to the error state";

  let description = [{
    This operation sets all kernel outputs to the error state. A kernel function
    can call set_error only once, and must not set any of the outputs in this
    case. The provided error message may be used by a runtime to propagate the
    error to the user.

    Example:

      ```mlir
      func @compute(%ctx: !rt.kernel_context) {
        %precondition = arith.cmpi ...
        cond_br %precondition, ^ok, ^err

      ^ok:
        %result = "compute_result"(): () -> memref<?xf32>
        rt.set_output %ctx, 0, %result : memref<?xf32>
        return

      ^err:
        rt.set_error %ctx, "Failed precondition"
        return
      }
      ```
  }];

  let arguments = (ins
    KernelContextType:$ctx,
    StrAttr:$error);

  let assemblyFormat = "$ctx `,` $error attr-dict";
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

def CustomCallOp : RT_Op<"custom_call"> {
  let summary = "calls a custom function registered with the runtime";

  let description = [{
    This operation calls a custom function registered with the runtime. This
    mechanism allows to call any C++ function from the compiled kernel running
    on top of the JitRt, for example this can be used as an extension mechanism
    to register vendor specific kernels (e.g. call oneDNN convolution).

    Example:

      ```mlir
      func @compute(%ctx: !rt.kernel_context, %arg0: memref<?xf32>,
                                              %arg1: memref<?xf32>) {
        %0 = rt.custom_call "one_dnn.some_operation"(%arg0, %arg1)
          : (memref<?xf32>, memref<?xf32>) -> !one_dnn.status
        return
      }
      ```

    To avoid collisions users should group custom calls into libraries and put
    them into namespaces (similar to MLIR dialects). In this example there is
    an assumption that all OneDnn related custom calls will be registered with
    a `one_dnn` prefix.
  }];

  let arguments = (ins
    StrAttr:$callee,
    Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

#endif // RT_OPS
