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

def GPU_CclIdType : GPU_Type<"CclId"> { let mnemonic = "ccl.id"; }

def GPU_CclDataTypeAttr : GPU_WrapperAttr<"CclDataType"> {
  let returnType = "::ncclDataType_t";
}
def GPU_CclReductionOpAttr : GPU_WrapperAttr<"CclReductionOp"> {
  let returnType = "::ncclRedOp_t";
}

def GPU_CclUniqueIdOp : GPU_Op<"ccl.unique_id"> {
  let description = [{
    tfrt_gpu.ccl.unique_id returns a !tfrt_gpu.ccl.id to be passed to
    tfrt_gpu.ccl.create. It encodes the machine's network address.
  }];
  let arguments = (ins GPU_PlatformAttr:$platform);
  let results = (outs GPU_CclIdType);
  let assemblyFormat = "custom<Enum>($platform) attr-dict";
}

def GPU_CclCreateOp : GPU_Op<"ccl.create"> {
  let description = [{
    tfrt_gpu.ccl.create returns a !tfrt_gpu.ccl.handle, which can be populated
    with individual collective ops (e.g., using tfrt_gpu.ccl.all_reduce) to be
    executed as a single fused operation.
  }];
  let arguments = (ins GPU_ContextType:$ctx, I32:$rank, I32:$count,
                   GPU_CclIdType:$id);
  let results = (outs GPU_CclHandleType);
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
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,` $chain
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
    custom<Enum>($reduction_op)`,` $chain attr-dict
  }];
}

def GPU_CclReduceScatterOp : GPU_Op<"ccl.reduce_scatter"> {
  let description = [{
    tfrt_gpu.ccl.reduce_scatter inserts a reduce_scatter operation (reducing the
    data array in the input buffer using reduction_op, and writing the reduced
    result scattered over the GPUs' output buffer) into the ccl.handle.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   GPU_BufferType:$output, GPU_CclDataTypeAttr:$data_type,
                   GPU_CclReductionOpAttr:$reduction_op, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,`
    custom<Enum>($reduction_op)`,` $chain attr-dict
  }];
}

def GPU_CclSendOp : GPU_Op<"ccl.send"> {
  let description = [{
    tfrt_gpu.ccl.send inserts into the ccl.handle an operation that sends data
    at the input buffer to the peer device.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   I32:$peer, GPU_CclDataTypeAttr:$data_type,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $peer`,` custom<Enum>($data_type)`,` $chain attr-dict
  }];
}

def GPU_CclRecvOp : GPU_Op<"ccl.recv"> {
  let description = [{
    tfrt_gpu.ccl.recv inserts into the ccl.handle an operation that receives
    data from the peer device into the output buffer.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$output,
                   I32:$peer, GPU_CclDataTypeAttr:$data_type,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $output`,` $peer`,` custom<Enum>($data_type)`,` $chain attr-dict
  }];
}

// In a similar buffer-splitting situation, a potential tfrt_gpu.mem.view op
// could be used to split buffers into sub-blocks.
def GPU_CclSplitSendRecvOp : GPU_Op<"ccl.split_send_recv"> {
  let description = [{
    tfrt_gpu.ccl.split_send_recv inserts into the ccl.handle a series of
    operations that send data at the input buffer and receive data into the
    output buffer in a manner that is evenly split across all peers: Device (i)
    sends (j)th block of data to device (j) to be placed as (i)th block.
  }];
  let arguments = (ins GPU_CclHandleType:$handle, GPU_BufferType:$input,
                   GPU_BufferType:$output, GPU_CclDataTypeAttr:$data_type,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $input`,` $output`,` custom<Enum>($data_type)`,` $chain attr-dict
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
