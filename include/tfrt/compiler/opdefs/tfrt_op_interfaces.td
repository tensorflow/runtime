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

#ifndef TFRT_OP_INTERFACES
#define TFRT_OP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TFRT Operations (Kernels) Cost Function Interface.
//===----------------------------------------------------------------------===//

def TFRT_CostFunctionInterface : OpInterface<"CostFunctionInterface"> {
  let cppNamespace = "::tfrt::compiler";

  let description = [{
    This interface provides an information about the cost of executing operation
    at runtime to the TFRT compiler passes, so that TFRT can make scheduling
    decisions, e.g. should the operation be launched asynchronously if it is
    expensive, or can be launched inline in the current thread if it is always
    cheap.

    For example all metadata manipulation operations are always cheap, and they
    should never be executed asynchronously.

    See common implementations in the `tfrt_traits.h` file.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the cost of executing a TFRT operation.}],
      "int64_t", "cost", (ins)
    >,
  ];
}

#endif // TFRT_OP_INTERFACES
