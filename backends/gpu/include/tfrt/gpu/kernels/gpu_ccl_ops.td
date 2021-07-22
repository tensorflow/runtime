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

//===- gpu_ccl_ops.td -----------------------------------------------------===//
//
// Operation definitions for GPU CCL (collective communications library)
// functionality.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_CCL_OPS
#else
#define GPU_CCL_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GPU_CclDataTypeAttr : GPU_WrapperAttr<"CclDataType">;
def GPU_CclReductionOpAttr : GPU_WrapperAttr<"CclReductionOp">;

#endif  // GPU_CCL_OPS
