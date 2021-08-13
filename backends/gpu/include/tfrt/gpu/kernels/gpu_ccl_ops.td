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

def GPU_CclDataTypeAttr : GPU_WrapperAttr<"CclDataType"> {
  let returnType = "::ncclDataType_t";
}
def GPU_CclReductionOpAttr : GPU_WrapperAttr<"CclReductionOp"> {
  let returnType = "::ncclRedOp_t";
}

def GPU_CclAllGatherOp : GPU_Op<"ccl.all_gather"> {
  let description = [{
    tfrt_gpu.ccl.all_gather inserts an all-gather operation (gathering other
    GPUs' values at the input buffer to the output buffer) into the ccl.handle.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   GPU_BufferType:$output, GPU_CclDataTypeAttr:$data_type,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,` $chain`,`
    attr-dict
  }];
}

def GPU_CclAllReduceOp : GPU_Op<"ccl.all_reduce"> {
  let description = [{
    tfrt_gpu.ccl.all_reduce inserts an all-reduce operation (reducing the data
    array in the input buffer using reduction_op, and writing the result in each
    output buffer) into the ccl.handle.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   GPU_BufferType:$output, GPU_CclDataTypeAttr:$data_type,
                   GPU_CclReductionOpAttr:$reduction_op, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,`
    custom<Enum>($reduction_op)`,` $chain`,` attr-dict
  }];
}

def GPU_CclExecuteOp : GPU_Op<"ccl.execute"> {
  let description = [{
    tfrt_gpu.ccl.execute runs the collective ops in the ccl.handle in a
    single fused group call.
  }];
  let arguments = (ins GPU_StreamType:$stream, GPU_CclHandleType:$handle,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
}

#endif  // GPU_CCL_OPS
