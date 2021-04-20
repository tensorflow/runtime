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

//===- cpurt_ops.td -------------------------------------------------------===//
//
// Operation definitions for CPU Runtime dialect.
//
//===----------------------------------------------------------------------===//

#ifdef CPURT_OPS
#else
#define CPURT_OPS

include "tfrt/basic_kernels/opdefs/tfrt_base.td"
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/tensor/opdefs/tensor.td"
include "tfrt/cpu/jit/opdefs/cpurt_base.td"

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime op definitions
//===----------------------------------------------------------------------===//

class CPURT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<TFRT_CPURT_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime op definitions to facilitate runtime lowerings.
//===----------------------------------------------------------------------===//

def CallOp : CPURT_Op<"call"> {
  let summary = "cpurt function call operation";
  let description = [{
    This operation is used in graph rewrites to call compiled functions. This
    operation is expanded into the pair of compile/execute operations during
    lowering (depending on execution mode it can be graph or corert operations).
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];

  // TODO(ezhulenev): Add a verifier for callee.
}

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime op definitions for native graph execution.
//===----------------------------------------------------------------------===//

def CompileOp : CPURT_Op<"compile"> {
  let summary = "cpurt compile operation";
  let description = [{
    `cpurt.compile` compiles a mlir CPU program to the cpurt JIT executable
    using LLVM JIT APIs and returns a cached `!cpurt.jit_executable` result.

    JIT executables support limited function signatures:

    1. Arguments must be ranked memrefs, and execute kernel converts dense
       tensors to memrefs when calls compiled kernels.

    2. Return values must be !async.token or !async.value of ranked memrefs.
       Async tokens converted to !tfrt.chain values, and async memrefs converted
       to dense host tensors.

       CoreRT execute operation also converts returned tensors into the CoreRT
       tensor handles.

    Example:

    ```mlir
      module @kernel attributes { tfrt.compiled } {
        func @main(...) {
        }
      }

      %executable = cpurt.compile { kernel = @kernel::@main }
    ```
  }];

  let arguments = (ins SymbolRefAttr:$kernel);
  let results = (outs JitExecutableType);
  let assemblyFormat = "attr-dict";
}



def ExecuteOp : CPURT_Op<"execute"> {
  let summary = "cpurt execute operation";
  let description = [{
    `cpurt.execute` takes an executable produced by the `cpurt.compile`
    operation and executes it with provided operands. It takes an async chain as
    a dependency (all operands are ready).

    Values returned from the compiled function converted to TFRT types:
      1. `!async.token` converted to `!tfrt.chain`
      2. `!async.value<memref<...>>` (or `memref<...>`) converted to `!t.tensor`

    Example:

    ```mlir
      // In bufferized program same buffers could be shared by multiple
      // operations, and their execution must be serialized with chains.
      %buffers_ready = "some_other_opertion"(%tensor0, %tensor1):
                       (!t.tensor, !t.tensor) -> !tfrt.chain

      // Wait for the completion of previous operation and execute compiled
      // kernel.
      %executed = cpurt.execute %executable [%buffers_ready]
                  (%tensor0, %tensor1) : (!t.tensor, !t.tensor) -> !tfrt.chain
    ```
  }];

  let arguments = (ins
    JitExecutableType:$jit_executable,
    TFRT_ChainType:$in_chain,
    Variadic<TensorType>:$operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $jit_executable `[` $in_chain `]` `(` $operands `)` attr-dict `:`
    functional-type($operands, $results)
  }];
}

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime op definitions for CoreRT interop.
//===----------------------------------------------------------------------===//

def CoreRtCompileOp : CPURT_Op<"corert.compile"> {
  let summary = "cpurt compile operation with CoreRT interop";
  let description = [{
    `cpurt.corert.compile` is almost identical to the cpurt.compile op, it only
    does additional compiled kernel verification to ensure that returned types
    could be converted to TensorHandles (return values must be async values of
    ranked memref type, async tokens are not supported).
  }];

  let arguments = (ins SymbolRefAttr:$kernel);
  let results = (outs JitExecutableType);
  let assemblyFormat = "attr-dict";
}

def CoreRtExecuteOp : CPURT_Op<"corert.execute"> {
  let summary = "cpurt execute operation with CoreRT interop";
  let description = [{
    `cpurt.corert.execute` provides integration with the CoreRT by taking
    `!corert.tensorhandle` operands and automatically converting all returned
    tensors to the CoreRT tensor handles, so that the user program does not
    have to do explicit `handle <-> tensor` conversions.
  }];

  let arguments = (ins
    JitExecutableType:$jit_executable,
    Variadic<CoreRT_TensorHandleType>:$operands
  );

  let results = (outs
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let assemblyFormat = [{
    $jit_executable `(` $operands `)` attr-dict `:`
    functional-type($operands, $results)
  }];
}

#endif // CPURT_OPS
