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

//===- tensor.td ----------------------------------------------------===//
//
// Operation definitions for generic TFRT tensor dialect.
//
//===----------------------------------------------------------------------===//

#ifdef TFRT_TENSOR_OPDEFS_TENSOR_H_
#else
#define TFRT_TENSOR_OPDEFS_TENSOR_H_

include "mlir/IR/OpBase.td"

// TFRT tensor dialect.
// TODO(b/170246041): Move `TensorType` under the TFRT dialect.
def Tensor_Dialect : Dialect {
  let name = "t";

  let description = [{
    The TFRT tensor dialect.

    This dialect defines operations to work with tensor in tfrt.
  }];

  let cppNamespace = "::tfrt::t";
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
def TensorType : DialectType<Tensor_Dialect,
    CPred<"$_self.isa<::tfrt::t::TensorType>()">, "!t.tensor type">,
    BuildableType<"$_builder.getType<::tfrt::t::TensorType>()"> {
  let description = [{
    `!t.tensor type` represents a generic tfrt tensor.
  }];
}

#endif  // TFRT_TENSOR_OPDEFS_TENSOR_H_
