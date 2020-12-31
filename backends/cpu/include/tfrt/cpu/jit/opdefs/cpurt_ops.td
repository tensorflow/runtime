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
include "tfrt/tensor/opdefs/tensor.td"

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime dialect definitions
//===----------------------------------------------------------------------===//

def TFRT_CPURT_Dialect : Dialect {
  let name = "cpurt";

  let description = [{
    The CPU runtime dialect.

    This dialect contains common CPU runtime operations.
  }];

  let cppNamespace = "::tfrt::cpu::jit";
}

def CompilationResultType : OpaqueType<"cpurt", "compilation_result",
                                       "!cpurt.compilation_result type">;

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime op definitions
//===----------------------------------------------------------------------===//

class CPURT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<TFRT_CPURT_Dialect, mnemonic, traits> {
}

def CompileOp : CPURT_Op<"compile"> {
  let summary = "cpurt compile operation";
  let description = [{
    cpurt.compile compiles a mlir CPU program to the executable binary format
    using LLVM JIT APIs and returns a cached compilationresult.

    Example:

    ```mlir
      %compilation_result = cpurt.compile { mlir_module =
        "<serialized mlir module with a kernel entrypoint function>"
      }
    ```
  }];

  let arguments = (ins StrAttr:$mlir_module);
  let results = (outs CompilationResultType);
  let assemblyFormat = "attr-dict";
}

def ExecuteOp : CPURT_Op<"execute"> {
  let summary = "cpurt execute operation";
  let description = [{
    cpurt.execute executes compiled binary produced by the cpurt.compiled
    operation with provided operands. It takes an async chain as a dependency
    (all operands are ready), and returns and async chain that signals execution
    completion (all results are ready).

    Example:

    ```mlir
      // In bufferized program same buffers could be shared by multiple
      // operations, and their execution must be serialized with chains.
      %buffers_ready = "some_other_opertion"(%tensor0, %tensor1):
                       (!t.tensor, !t.tensor) -> !tfrt.chain

      // Wait for the completion of previous operation and execute compiled
      // kernel.
      %executed = cpurt.execute %compilation_result[%buffers_ready]
                  (%tensor0, %tensor1) : !t.tensor, !t.tensor
    ```
  }];

  let arguments = (ins
    CompilationResultType:$compilation_result,
    TFRT_ChainType:$in_chain,
    Variadic<TensorType>:$operands
  );

  let results = (outs
    TFRT_ChainType:$out_chain
  );

  let assemblyFormat = [{
    $compilation_result `[` $in_chain `]` `(` $operands `)` attr-dict `:` type($operands)
  }];
}

#endif // CPURT_OPS
