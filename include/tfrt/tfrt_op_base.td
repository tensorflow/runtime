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
include "tfrt/basic_kernels/opdefs/tfrt_base.td"
include "tfrt/tensor/opdefs/host_tensor.td"
include "tfrt/tensor/opdefs/tensor.td"

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//


#endif  // TFRT_OP_BASE
