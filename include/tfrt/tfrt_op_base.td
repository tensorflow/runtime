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

//===-- tfrt_op_base.td - Base op definition file ----------*- tablegen -*-===//
//
// This is the base operation definition file for TFRT ops.
//
//===----------------------------------------------------------------------===//

#ifdef TFRT_OP_BASE
#else
#define TFRT_OP_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
def ChainType : OpaqueType<"hex", "chain", "!hex.chain type">;
def StringType : OpaqueType<"hex", "string", "!hex.string type">;

def TensorType
  : OpaqueType<"t", "tensor", "tensor">;

def HostBufferType
  : OpaqueType<"ht", "host_buffer", "host_buffer">;
def TensorHandleType
  : OpaqueType<"corert", "tensorhandle", "!corert.tensorhandle type">;

#endif  // TFRT_OP_BASE
