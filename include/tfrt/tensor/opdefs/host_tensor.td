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

//===- host_tensor.td ----------------------------------------------------===//
//
// Operation definitions for host tensor dialect.
//
//===----------------------------------------------------------------------===//

#ifdef TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_
#else
#define TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_

include "mlir/IR/OpBase.td"

// TFRT host tensor dialect.
// TODO(b/170246041): Consider move this into the TFRT dialect instead.
def HostTensor_Dialect : Dialect {
  let name = "ht";

  let description = [{
    The host tensor dialect.

    This dialect defines operations to work with host tensor in tfrt.
  }];

  let cppNamespace = "::tfrt::ht";
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
def HostBufferType : DialectType<HostTensor_Dialect,
    CPred<"$_self.isa<::tfrt::ht::HostBufferType>()">, "!ht.host_buffer type">,
    BuildableType<"$_builder.getType<::tfrt::ht::HostBufferType>()"> {
  let description = [{
    `!ht.host_buffer type` represents a host buffer.
  }];
}

#endif  // TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_
