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

//===- cuda_ops.td ---------------------------------------------------------===//
//
// CUDA based CUDA pperation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_OPS
#else
#define CUDA_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_blas_ops.td"

def InitOp : CUDA_Op<"init"> {
  let summary = "cuda init operation";
  let description = [{
    cuda.init initializes the underlying CUDA driver API.

    Must be called before all other CUDA operations.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.init %ch1
  }];
  let arguments = (ins TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def DeviceGetOp : CUDA_Op<"device.get"> {
  let summary = "cuda device.get operation";
  let description = [{
    cuda.device.get returns the CUDA Device at the given index.

    Example:
      %ch1 = tfrt.new.chain
      %runtime, %ch2 = cuda.init %ch1
      %index = tfrt.constant.i32 0
      %device, %ch3 = cuda.device.get %index, %ch2
  }];
  let arguments = (ins I32, TFRT_ChainType);
  let results = (outs DeviceType, TFRT_ChainType);
}

def StreamCreateOp : CUDA_Op<"stream.create"> {
  let summary = "cuda stream.create operation";
  let description = [{
    cuda.stream.create creates a CUDA stream in the given context.

    Created stream does not perform implicit synchronization with stream 0.

    Example:
      %ch1 = tfrt.new.chain
      %stream, %ch2 = cuda.stream.create %context, %ch1
  }];
  let arguments = (ins ContextType, TFRT_ChainType);
  let results = (outs StreamType, TFRT_ChainType);
}

def StreamSynchronizeOp : CUDA_Op<"stream.synchronize"> {
  let summary = "cuda stream.synchronize operation";
  let description = [{
    cuda.stream.synchronize waits until all stream's tasks are completed.

    This op will block caller thread, and is not intended for use in production.
    It is only intended for tests and benchmarks. Production users must use cuda
    events (see `cuda.event.*` ops defined below).

    Example:
      %synced = cuda.stream.synchronize %stream, %ch
  }];
  let arguments = (ins StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def EventCreateOp : CUDA_Op<"event.create"> {
  let summary = "cuda event.create operation";
  let description = [{
    cuda.event.create creates a CUDA event.

    Example:
      %ch1 = tfrt.new.chain
      %event, %ch2 = cuda.event.create %context %ch1
  }];
  let arguments = (ins ContextType, TFRT_ChainType);
  let results = (outs EventType, TFRT_ChainType);
}

def EventRecordOp : CUDA_Op<"event.record"> {
  let summary = "cuda event.record operation";
  let description = [{
    cuda.event.record records a CUDA event on the given stream.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.event.record %event, %stream, %ch1
  }];
  let arguments = (ins EventType, StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def EventPollOp : CUDA_Op<"event.poll"> {
  let summary = "cuda event.poll operation";
  let description = [{
    cuda.event.poll polls for completion of work recorded in the given event.

    This op will set the returned chain when the event has been reached.
    An example implementation is to call cuEventQuery on the given
    event periodically.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.event.poll %event %ch1
  }];
  let arguments = (ins EventType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def AllocatorCreateOp : CUDA_Op<"allocator.create"> {
  let summary = "cuda allocator.create operation";
  let description = [{
    cuda.allocator.create creates an allocator for the given context.

    Example:
      %ch1 = tfrt.new.chain
      %allocator, %ch2 = cuda.allocator.create %context, %ch1
  }];
  let arguments = (ins ContextType, TFRT_ChainType);
  let results = (outs AllocatorType, TFRT_ChainType);
}

def AllocatorDestroyOp : CUDA_Op<"allocator.destroy"> {
  let summary = "cuda allocator.destroy operation";
  let description = [{
    cuda.allocator.destroy destroys an allocator.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.allocator.destroy %allocator, %ch1
  }];
  let arguments = (ins AllocatorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def MemAllocateOp : CUDA_Op<"mem.allocate"> {
  let summary = "cuda allocate operation";
  let description = [{
    cuda.mem.allocate allocates a buffer of device memory.

    Allocation is associated with a "primary" stream. For best performance,
    the allocated buffer should be used primarily on the primary stream.
    Usage on other streams is permitted, but users must synchronize streams
    appropriately. For example, if a kernel on stream1 writes to the buffer
    and a kernel on stream2 reads from it, users must synchronize the streams
    to make sure the read happens after the write.

    Moreover, users must always synchronize the first use of the
    buffer on a non-primary stream to the primary stream (at the time of
    allocation). Even when the buffer is not used on the primary stream or
    when both accesses are reads or writes. For example, the following usage
    pattern will result in undefined behavior:

      %buf, %ch1 = cuda.mem.allocate %stream1, %size, %ch0
      cuda.launch %stream1, %kernel_reading_buf, %buf, %ch1
      cuda.launch %stream2, %another_kernel_reading_buf, %buf, %ch1

    Users must add synchronization to make sure use on stream2 happens after
    everything that was on stream1, at the time of allocation, has finished, e.g.

      %buf, %ch1 = cuda.mem.allocate %stream1, %size, %ch0
      %event, %ch2 = cuda.event.create %ch1
      %ch3 = cuda.event.record %stream1, %event, %ch2
      cuda.launch %stream1, %kernel_reading_buf, %buf
      %ch4 = cuda.event.wait %stream2, %event, %ch3
      cuda.launch %stream2, %another_kernel_reading_buf, %buf, %ch4

    Example:
      %ch0 = tfrt.new.chain
      %buffer, %ch1 = cuda.mem.allocate %stream, %size, %ch0
  }];
  let arguments = (ins AllocatorType, StreamType, I64:$size, TFRT_ChainType);
  let results = (outs BufferType, TFRT_ChainType);
}

def MemPrintOp : CUDA_Op<"mem.print_metadata"> {
  let summary = "cuda mem.print operation";
  let description = [{
    cuda.mem.print_metadata prints a CUDA buffer metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.mem.print_metadata %buffer %ch1
  }];
  let arguments = (ins BufferType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

class TensorMakeOp<string dtype> : CUDA_Op<"tensor.make." # dtype> {
  let summary = "cuda tensor.make operation";
  let description = [{
    cuda.tensor.make makes a tensor from the given buffer

    The size of the buffer must match the size needed to hold the tensor,
    i.e. the number of elements, of requested dtype, in the given shape.

    Example:
      %ch0 = tfrt.new.chain
      %buffer, %ch1 = cuda.mem.allocate %stream, %size, %ch0
      %shape = ts.build_shape [2 : i32, 4 : i32]
      %tensor, %ch2 = cuda.tensor.make.f32 %buffer %shape %ch1
  }];
  let arguments = (ins BufferType, TS_Shape, TFRT_ChainType);
  let results = (outs TensorType, TFRT_ChainType);
}

foreach dtype = ["i8", "i32", "i64", "f32", "f64"] in {
  def CUDA_TensorMakeOp_#dtype : TensorMakeOp<dtype>;
}

def TensorPrintOp : CUDA_Op<"tensor.print_metadata"> {
  let summary = "cuda tensor.print_metadata operation";
  let description = [{
    cuda.tensor.print prints a CUDA tensor metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.tensor.print_metadata %tensor, %ch1
  }];
  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def MemcpyHtoDOp : CUDA_Op<"mem.copy_host_to_device"> {
  let summary = "cuda mem.copy_host_to_device operation";
  let description = [{
    cuda.mem.copy_host_to_device copies memory from host to device.

    At this time, the user must make sure that host buffer is not deleted
    until the copy completes.
    TODO(iga): Extend the life automatically

    Example:
      %ch1 = cuda.mem.copy_host_to_device %ctx, %dst_buf, %src_buf, %count_bytes, %stream, %ch0
  }];
  let arguments = (ins ContextType, BufferType:$dst, HostBufferType:$src, I64, StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def MemcpyDtoHOp : CUDA_Op<"mem.copy_device_to_host"> {
  let summary = "cuda mem.copy_device_to_host operation";
  let description = [{
    cuda.mem.copy_device_to_host copies memory from device to host.

    At this time, the user must make sure that host buffer is not deleted
    until the copy completes. This should happen naturally since the user
    generally does something with the destination host buffer after the copy.
    TODO(iga): Add extend the life automatically

    Example:
      %ch1 = cuda.mem.copy_device_to_host %ctx, %dst_buf, %src_buf, %count_bytes, %stream, %ch0
  }];
  let arguments = (ins ContextType, HostBufferType:$dst, BufferType:$src, I64, StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

#endif  // CUDA_OPS
