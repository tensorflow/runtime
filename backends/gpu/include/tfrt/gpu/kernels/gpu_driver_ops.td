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

//===- gpu_ops.td --------------------------------------------------------===//
//
// tfrt_gpu dialect operation definitions.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_DRIVER_OPS
#else
#define GPU_DRIVER_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"

def GPU_DeviceType : GPU_Type<"device">;
def GPU_EventType : GPU_Type<"event">;
def GPU_FunctionType : GPU_Type<"function">;

def GPU_DeviceGetOp : GPU_Op<"device.get"> {
  let description = [{
    tfrt_gpu.device.get returns the gpu device at the given index.

    Example:
      %ch1 = tfrt.new.chain
      %index = tfrt.constant.i32 0
      %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : ui32 }
  }];
  let arguments = (ins I32, TFRT_ChainType, I32Attr:$platform);
  let results = (outs GPU_DeviceType);
}

def GPU_ContextCreateOp : GPU_Op<"context.create"> {
  let description = [{
    tfrt_gpu.context.create returns a gpu context for the given device.

    Example:
      %ch1 = tfrt.new.chain
      %index = tfrt.constant.i32 0
      %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : ui32 }
      %ctx = tfrt_gpu.context.create %device, %ch1
  }];
  let arguments = (ins GPU_DeviceType, TFRT_ChainType);
  let results = (outs GPU_ContextType);
}

def GPU_StreamCreateOp : GPU_Op<"stream.create"> {
  let description = [{
    tfrt_gpu.stream.create creates a gpu stream in the given context.

    Created stream does not perform implicit synchronization with stream 0.

    Example:
      %ch1 = tfrt.new.chain
      %stream = tfrt_gpu.stream.create %context, %ch1
  }];
  let arguments = (ins GPU_ContextType, TFRT_ChainType);
  let results = (outs GPU_StreamType);
}

def GPU_StreamSynchronizeOp : GPU_Op<"stream.synchronize"> {
  let description = [{
    tfrt_gpu.stream.synchronize waits until all the work scheduled on the
    stream has completed.

    This op will set the returned chain when the stream has been synchronized.

Example:
      %synced = tfrt_gpu.stream.synchronize %stream, %ch
  }];
  let arguments = (ins GPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_EventCreateOp : GPU_Op<"event.create"> {
  let description = [{
    tfrt_gpu.event.create creates a gpu event in the provided context.

    Example:
      %event = tfrt_gpu.event.create %context
  }];
  let arguments = (ins GPU_ContextType, TFRT_ChainType);
  let results = (outs GPU_EventType);
}

def GPU_EventRecordOp : GPU_Op<"event.record"> {
  let description = [{
    tfrt_gpu.event.record records a gpu event on the given stream.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.event.record %event, %stream, %ch1
  }];
  let arguments = (ins GPU_EventType, GPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_EventPollOp : GPU_Op<"event.synchronize"> {
  let description = [{
    tfrt_gpu.event.synchronize waits for the completion of work captured by the
    event.

    This op will set the returned chain when the event has been synchronized.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.event.synchronize %event %ch1
  }];
  let arguments = (ins GPU_EventType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_AllocatorCreateOp : GPU_Op<"allocator.create"> {
  let description = [{
    tfrt_gpu.allocator.create creates an allocator for the given context.

    Example:
      %ch1 = tfrt.new.chain
      %allocator = tfrt_gpu.allocator.create %context, %ch1
  }];
  let arguments = (ins GPU_ContextType, TFRT_ChainType);
  let results = (outs GPU_AllocatorType);
}

def GPU_MemAllocateOp : GPU_Op<"mem.allocate"> {
  let description = [{
    tfrt_gpu.mem.allocate allocates a buffer of device memory.

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

      %buf = tfrt_gpu.mem.allocate %stream1, %size, %ch0
      tfrt_gpu.launch %stream1, %kernel_reading_buf, %buf, %ch0
      tfrt_gpu.launch %stream2, %another_kernel_reading_buf, %buf, %ch0

    Users must add synchronization to make sure use on stream2 happens after
    everything that was on stream1, at the time of allocation, has finished, e.g.

      %buf = tfrt_gpu.mem.allocate %stream1, %size, %ch0
      %event = tfrt_gpu.event.create %ch0
      %ch2 = tfrt_gpu.event.record %stream1, %event, %ch1
      tfrt_gpu.launch %stream1, %kernel_reading_buf, %buf, %ch0
      %ch3 = tfrt_gpu.stream.wait %stream2, %event, %ch2
      tfrt_gpu.launch %stream2, %another_kernel_reading_buf, %buf, %ch3

    Example:
      %ch0 = tfrt.new.chain
      %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0
  }];
  let arguments = (ins GPU_AllocatorType, GPU_StreamType, I64:$size, TFRT_ChainType);
  let results = (outs GPU_BufferType);
}

def GPU_MemPrintOp : GPU_Op<"mem.print_metadata"> {
  let description = [{
    tfrt_gpu.mem.print_metadata prints a gpu buffer metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.mem.print_metadata %buffer %ch1
  }];
  let arguments = (ins GPU_BufferType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

class TensorMakeOp<string dtype> : GPU_Op<"tensor.make." # dtype> {
  let description = [{
    tfrt_gpu.tensor.make makes a tensor from the given buffer

    The size of the buffer must match the size needed to hold the tensor,
    i.e. the number of elements, of requested dtype, in the given shape.

    Example:
      %ch0 = tfrt.new.chain
      %buffer, %ch1 = tfrt_gpu.mem.allocate %stream, %size, %ch0
      %shape = ts.build_shape [2 : i32, 4 : i32]
      %tensor, %ch2 = tfrt_gpu.tensor.make.f32 %buffer %shape %ch1
  }];
  let arguments = (ins GPU_BufferType, TS_Shape, TFRT_ChainType);
  let results = (outs TensorType);
}

// TODO(csigg): this should just be a type attribute.
foreach dtype = ["i8", "i32", "i64", "f32", "f64"] in {
  def GPU_TensorMakeOp_#dtype : TensorMakeOp<dtype>;
}

def GPU_TensorPrintOp : GPU_Op<"tensor.print_metadata"> {
  let description = [{
    tfrt_gpu.tensor.print prints a gpu tensor metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.tensor.print_metadata %tensor, %ch1
  }];
  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_MemcpyHtoDOp : GPU_Op<"mem.copy_host_to_device"> {
  let description = [{
    tfrt_gpu.mem.copy_host_to_device copies memory from host to device.

    At this time, the user must make sure that host buffer is not deleted
    until the copy completes.
    TODO(iga): Extend the life automatically

    Example:
      %ch1 = tfrt_gpu.mem.copy_host_to_device %ctx, %dst_buf, %src_buf, %count_bytes, %stream, %ch0
  }];
  let arguments = (ins GPU_BufferType:$dst, HostBufferType:$src,
                   I64:$size_bytes, GPU_StreamType:$stream, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_MemcpyDtoHOp : GPU_Op<"mem.copy_device_to_host"> {
  let description = [{
    tfrt_gpu.mem.copy_device_to_host copies memory from device to host.

    At this time, the user must make sure that host buffer is not deleted
    until the copy completes. This should happen naturally since the user
    generally does something with the destination host buffer after the copy.
    TODO(iga): Add extend the life automatically

    Example:
      %ch1 = tfrt_gpu.mem.copy_device_to_host %ctx, %dst_buf, %src_buf, %count_bytes, %stream, %ch0
  }];
  let arguments = (ins HostBufferType:$dst, GPU_BufferType:$src,
                   I64:$size_bytes, GPU_StreamType:$stream, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_FunctionLoadOp : GPU_Op<"function.load"> {
  let description = [{
    tfrt_gpu.function.load loads GPU device function from a binary blob.

    The function is cached in the context for better performance. The key needs
    to be unique across all modules loaded into a context.

    Both data and name attribute must be explicitly null-terminated.

    Example:
      %function = tfrt_gpu.function.load %ctx, %chain {
        data = "null-terminated device function blob, e.g. PTX",
        key = 1234,
        name = "kernel\00",
      }
  }];
  let arguments = (ins
    GPU_ContextType:$ctx,
    TFRT_ChainType:$in_chain,
    StrAttr:$data,
    UI64Attr:$key,
    StrAttr:$name
  );
  let results = (outs GPU_FunctionType);
}

def GPU_LaunchOp : GPU_Op<"function.launch"> {
  let description = [{
    tfrt_gpu.function.launch invokes the provided device function on a stream.

    Example:
      %ch1 = tfrt_gpu.function.launch %stream, %function,
                 blocks in (%grid_dim_x %grid_dim_y %grid_dim_z),
                 threads in (%block_dim_x %block_dim_y %block_dim_z),
                 %shared_memory_size_bytes, %chain,
                 args(%arg0, %arg1, ..., %argN) : (<type0>, <type1>, ... <typeN>)
  }];
  let arguments = (ins
    GPU_StreamType:$stream,
    GPU_FunctionType:$function,
    UI32:$grid_dim_x,
    UI32:$grid_dim_y,
    UI32:$grid_dim_z,
    UI32:$block_dim_x,
    UI32:$block_dim_y,
    UI32:$block_dim_z,
    UI32:$shared_memory_size_bytes,
    TFRT_ChainType:$in_chain,
    Variadic<AnyTypeOf<[GPU_BufferType, AnyFloat, AnyInteger]>>:$args
  );
  let results = (outs TFRT_ChainType:$out_chain);
  let assemblyFormat = [{
    $stream`,` $function`,`
    `blocks` `in` ` ` `(`$grid_dim_x`,` $grid_dim_y`,` $grid_dim_z`)``,`
    `threads` `in` ` ` `(`$block_dim_x`,`  $block_dim_y`,`  $block_dim_z`)``,`
    $shared_memory_size_bytes`,` $in_chain
    ( `,` `args``(` $args^ `)`  `:` `(` type($args) `)`  )? attr-dict
  }];
}

#endif  // GPU_OPS
