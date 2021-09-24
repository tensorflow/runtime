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

//===- rt_base.td ------------------------------------------------------===//
//
// Dialect and type definitions for the RT dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_BASE
#else
#define RT_BASE

include "mlir/IR/OpBase.td"

def RuntimeDialect : Dialect {
  let name = "rt";

  let description = [{
    The run-time dialect.

    This dialect contains operations to interact with the TFRT run-time.
  }];

  let cppNamespace = "::tfrt::cpu::jit";
}

class RT_Type<string name, string typeMnemonic> : TypeDef<RuntimeDialect,
                                                          name> {
  let mnemonic = typeMnemonic;
}

// This is an opaque handle to tfrt::cpu::jit::KernelContextType.
def KernelContextType : RT_Type<"RuntimeKernelContext",
                                "kernel_context"> {
  let summary = "Kernel Context type";
  let description = [{
    Opaque handle used for interacting with the TFRT run-time.
  }];
}

#endif // RT_BASE
