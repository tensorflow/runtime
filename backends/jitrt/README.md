# TFRT JitRt backend

JitRt is a set of libraries that helps to write the compile and execute kernels
on top of the TFRT runtime and MLIR compiler infrastructure.

## Example

```
module @rsqrt_m attributes { tfrt.compiled } {
  func @rsqrt(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Rsqrt"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

func @compute(%arg0: !tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor {
  %0 = tf_jitrt.fallback.execute @rsqrt_m::@rsqrt(%arg0) device("/device:CPU:0")
         : (!tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor
  return %0 : !tfrt_fallback.tf_tensor
}
```

The nested module marked with `tfrt.compiled` compiled at runtime to the native
X86 binary using an MLIR pipeline that lowers from Tensorflow dialect to LLVM,
and then compiled to native executable using LLVM ORC APIs.

The `@compute` function is lowered to BEF function, and interpreted by the
BefExecutor.

## Features

### Configurable compilation and specialization pipelines

Input IR can be in any dialect (Tensorflow, MHLO, etc...), and client of the
JitRt configures the execution by passing compilation pipelines. For example in
Autofusion/TFRT the client is Tensorflow (Servomatic), and the compilation
pipeline is constructed to lower Tensorflow programs to LLVM.

### Shape and Value specializations and operands constraints

JitRt supports input IR shape and value specialization, e.g. if some ranks,
shapes or values are required to compile the binary, JitRt will automatically
specialize the original IR and recompile at runtime.

Compiled function can define constraints on its inputs, that must be resolved
before the function can be compiled. If constraints can't be resolved statically
from the function signature (e.g. rank is unknown), then the runtime will
specialize generic function to concrete operands at runtime (concrete operands
rank, shape or value).

If function inputs do not have unresolved constraints, compiler will instantiate
the default executable, that can take all compatible inputs without
recompilation.

(a) Rank constraint:

```
%arg : tensor<*xf32> { rt.constraint = "rank" }
```

Before compiling the function, unranked input type will be updated to the
corresponding ranked input type (e.g. unranked tensor -> ranked tensor).

(b) Shape constraint:

```
%arg : tensor<?x?xf32> { rt.constraint = "shape" }
```

Shape of the runtime argument will be used to specialize the compiled function,
if this shape seen the first time, it will trigger function recompilation.

(c) Value constraint:

```
%reduction_dimension : tensor<i32> { rt.constraint = "value" }
```

Runtime value will be sunk into the body of a function as a constant, and the
function will be recompiled. For example this can be used to sink reduction
dimensions to generate more efficient code.

Value constraint is only supported for the integer data type, in practice it
should be reduction dimension, dimension permutation, or any similar value that
does not change often, and is required for generating efficient code.

#### Shape and value specialization example:

```
// Computes %arg0 mean value over the axis specified by the %arg1.
// See: https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
func @mean(%arg0: tensor<?x?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
  %0 = "tf.Mean(%arg0, %arg1)
         : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xf32>
  return %0: tensor<?xf32>
}
```

#### Shape specialization to input shapes: [tensor<4x8xf32>, tensor<f32>]

```
   func @mean(%arg0: tensor<4x8xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
     %0 = "tf.Mean(%arg0, %arg1)
            : (tensor<4x8xf32>, tensor<i32>) -> tensor<?xf32>
     return %0: tensor<?xf32>
   }
```

Shape specialization in this particular case doesn't bring much improvement,
because without knowing the reduction axis we can't infer any new information
from the input shape alone.

#### Value specialization to input values: [<no-specialize>, dense<1 : i32>]

```
   func @mean(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
     %0 = "tf.Constant" { value = dense<1 : i32>} -> tensor<i32>
     %1 = "tf.Mean(%arg0, %0)
            : (tensor<4x8xf32>, tensor<i32>) -> tensor<4xf32>
     return %1 : tensor<4xf32>
   }
```

By specializing function to the concrete value of the second argument, by
sinking it into the function body we can infer the output shape. Also this
information allows to statically choose reduction implementation optimized for
reducing along the innermost dimension.

Furthermore static information about reduction axis allows to lower mean
operation to Linalg generic operation. Dynamic reduction axis is not
representable in Linalg, and would require multi-versioning and dynamic dispatch
at runtime.

### End To End Example

See `tfrt/backends/jitrt/cpp_tests/end_to_end_example_test.cc` test for and end
to end compilation and execution example using C++ APIs.
