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

#ifdef TFRTGPU_DRIVER_OPS
#else
#define TFRTGPU_DRIVER_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TFRTGPU_DeviceType : TFRTGPU_Type<"Device"> { let mnemonic = "device"; }
def TFRTGPU_EventType : TFRTGPU_Type<"Event"> { let mnemonic = "event"; }
def TFRTGPU_FunctionType : TFRTGPU_Type<"Function"> { let mnemonic = "function"; }

def TFRTGPU_DeviceGetOp : TFRTGPU_Op<"device.get", [Pure]> {
  let description = [{
    tfrt_gpu.device.get returns the gpu device at the given index.

    Example:
      %ordinal = tfrt.constant.i32 0
      %device = tfrt_gpu.device.get CUDA, %ordinal
  }];
  let arguments = (ins TFRTGPU_PlatformAttr:$platform, I32:$ordinal);
  let results = (outs TFRTGPU_DeviceType);
  let assemblyFormat = "custom<Enum>($platform)`,` operands attr-dict";
}

def TFRTGPU_ContextPrimaryOp : TFRTGPU_Op<"context.primary", [Pure]> {
  let description = [{
    tfrt_gpu.context.primary returns the primary gpu context of the device.

    Example:
      %ordinal = tfrt.constant.i32 0
      %device = tfrt_gpu.device.get CUDA, %ordinal
      %ctx = tfrt_gpu.context.primary %device
  }];
  let arguments = (ins TFRTGPU_DeviceType);
  let results = (outs TFRTGPU_ContextType);
}

def TFRTGPU_ContextCreateOp : TFRTGPU_Op<"context.create"> {
  let description = [{
    tfrt_gpu.context.create returns a gpu context for the given device.

    Example:
      %ordinal = tfrt.constant.i32 0
      %device = tfrt_gpu.device.get CUDA, %ordinal
      %ctx = tfrt_gpu.context.create %device
  }];
  let arguments = (ins TFRTGPU_DeviceType);
  let results = (outs TFRTGPU_ContextType);
}

def TFRTGPU_StreamCreateOp : TFRTGPU_Op<"stream.create"> {
  let description = [{
    tfrt_gpu.stream.create creates a gpu stream in the given context.

    Created stream does not perform implicit synchronization with stream 0.

    Example:
      %ch1 = tfrt.new.chain
      %stream = tfrt_gpu.stream.create %context
  }];
  let arguments = (ins TFRTGPU_ContextType);
  let results = (outs TFRTGPU_StreamType);
}

def TFRTGPU_StreamGetContextOp : TFRTGPU_Op<"stream.get_context", [Pure]> {
  let description = [{
    tfrt_gpu.stream.get_context returns the context the stream was created with.

    Example:
      %context = tfrt_gpu.stream.get_context %stream
  }];
  let arguments = (ins TFRTGPU_StreamType);
  let results = (outs TFRTGPU_ContextType);
}

def TFRTGPU_StreamWaitOp : TFRTGPU_Op<"stream.wait"> {
  let description = [{
    tfrt_gpu.stream.wait makes a stream wait on a gpu event.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.stream.wait %stream, %event, %ch1
  }];
  let arguments = (ins TFRTGPU_StreamType, TFRTGPU_EventType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_StreamSynchronizeOp : TFRTGPU_Op<"stream.synchronize"> {
  let description = [{
    tfrt_gpu.stream.synchronize waits until all the work scheduled on the
    stream has completed.

    This op will set the returned chain when the stream has been synchronized.

Example:
      %synced = tfrt_gpu.stream.synchronize %stream, %ch
  }];
  let arguments = (ins TFRTGPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_EventCreateOp : TFRTGPU_Op<"event.create"> {
  let description = [{
    tfrt_gpu.event.create creates a gpu event in the provided context.

    Example:
      %event = tfrt_gpu.event.create %context
  }];
  let arguments = (ins TFRTGPU_ContextType);
  let results = (outs TFRTGPU_EventType);
}

def TFRTGPU_EventRecordOp : TFRTGPU_Op<"event.record"> {
  let description = [{
    tfrt_gpu.event.record records a gpu event on the given stream.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.event.record %event, %stream, %ch1
  }];
  let arguments = (ins TFRTGPU_EventType, TFRTGPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_EventSynchronizeOp : TFRTGPU_Op<"event.synchronize"> {
  let description = [{
    tfrt_gpu.event.synchronize waits for the completion of work captured by the
    event.

    This op will set the returned chain when the event has been synchronized.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.event.synchronize %event, %ch1
  }];
  let arguments = (ins TFRTGPU_EventType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_AllocatorCreateOp : TFRTGPU_Op<"allocator.create", [Pure]> {
  let description = [{
    tfrt_gpu.allocator.create creates an allocator for the given context.

    Example:
      %allocator = tfrt_gpu.allocator.create %context
  }];
  let arguments = (ins TFRTGPU_ContextType);
  let results = (outs TFRTGPU_AllocatorType);
}

def TFRTGPU_MemAllocateOp : TFRTGPU_Op<"mem.allocate"> {
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

      %buf = tfrt_gpu.mem.allocate %allocator, %stream1, %size, %ch0
      %event = tfrt_gpu.event.create %ch0
      %ch2 = tfrt_gpu.event.record %stream1, %event, %ch1
      tfrt_gpu.launch %stream1, %kernel_reading_buf, %buf, %ch0
      %ch3 = tfrt_gpu.stream.wait %stream2, %event, %ch2
      tfrt_gpu.launch %stream2, %another_kernel_reading_buf, %buf, %ch3

    Example:
      %ch0 = tfrt.new.chain
      %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0
  }];
  let arguments = (ins TFRTGPU_AllocatorType, TFRTGPU_StreamType, I64:$size, TFRT_ChainType);
  let results = (outs TFRTGPU_BufferType);
}

def TFRTGPU_MemAllocateHostOp : TFRTGPU_Op<"mem.allocate_host"> {
  let description = [{
    tfrt_gpu.mem.allocate_host allocates a buffer of host memory.

    Example:
      %buffer = tfrt_gpu.mem.allocate_host %ctx, %size, %ch0
  }];
  let arguments = (ins TFRTGPU_ContextType, I64:$size, TFRT_ChainType);
  let results = (outs TFRTGPU_BufferType);
}

def TFRTGPU_MemDeallocateOp : TFRTGPU_Op<"mem.deallocate"> {
  let description = [{
    tfrt_gpu.mem.deallocate frees a buffer of device memory.

    Any use of the 'buffer' must be synchronized with 'stream' beforehand.

    Example:
      %ch0 = tfrt.new.chain
      %ch1 = tfrt_gpu.mem.deallocate %buffer, %stream, %ch0
  }];
  let arguments = (ins TFRTGPU_BufferType, TFRTGPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_MemCopyOp : TFRTGPU_Op<"mem.copy"> {
  let description = [{
    tfrt_gpu.mem.copy copies memory between host or device.

    Execution is asynchronous. Argument buffers on the host must not be deleted
    and are not guaranteed to be valid before the stream is synchronized with
    the host. If the source and destination are the same, the copy is skipped.

    Example:
      %ch1 = tfrt_gpu.mem.copy %dst, %src, %stream, %ch0
                 : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  }];
  let arguments = (ins AnyTypeOf<[TFRTGPU_BufferType, HostBufferType]>:$dst,
                   AnyTypeOf<[TFRTGPU_BufferType, HostBufferType]>:$src,
                   TFRTGPU_StreamType:$stream, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);

  let assemblyFormat = [{
    $dst`,` $src`,` $stream`,` $chain attr-dict `:` type($dst)`,` type($src)
  }];

}

def TFRTGPU_MemSetOp : TFRTGPU_Op<"mem.set"> {
  let description = [{
    tfrt_gpu.mem.set fills dst memory with a scalar value.

    Execution is asynchronous.

    Example:
      %ch1 = tfrt_gpu.mem.set %dst, %value, %stream, %ch0 : f32
  }];
  let arguments = (ins TFRTGPU_BufferType:$dst,
                   AnyType:$value,
                   TFRTGPU_StreamType:$stream, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);

  let assemblyFormat = [{
    $dst`,` $value`,` $stream`,` $chain attr-dict `:` type($value)
  }];

}

def TFRTGPU_MemRegisterOp : TFRTGPU_Op<"mem.register"> {
  let description = [{
    tfrt_gpu.mem.register page-locks a host buffer for device access.

    The lifetime of %buffer is extended by the lifetime of %result.

    Example:
      %result = tfrt_gpu.mem.register %ctx, %buffer
  }];
  let arguments = (ins TFRTGPU_ContextType:$ctx, HostBufferType:$buffer);
  let results = (outs TFRTGPU_BufferType);

  let assemblyFormat = "$ctx`,` $buffer attr-dict";
}

def TFRTGPU_MemViewOp : TFRTGPU_Op<"mem.view", [Pure]> {
  let description = [{
    tfrt_gpu.mem.view returns a buffer view of the given byte size and offset.

    The lifetime of %buffer is extended by the lifetime of %result.

    Example:
      %offset = tfrt.constant.ui64 42
      %size   = tfrt.constant.ui64 64
      %result = tfrt_gpu.mem.register %buffer, %offset, %size
  }];
  let arguments = (ins TFRTGPU_BufferType:$buffer, UI64:$offset, UI64:$size);
  let results = (outs TFRTGPU_BufferType);

  let assemblyFormat = "$buffer`,` $offset`,` $size attr-dict";
}

def TFRTGPU_MemLoadOp : TFRTGPU_Op<"mem.load"> {
  let description = [{
    tfrt_gpu.mem.load returns the value stored in a buffer of host memory.

    Example:
      %result = tfrt_gpu.mem.load %buffer, %ch : i32
  }];
  let arguments = (ins TFRTGPU_BufferType:$buffer, TFRT_ChainType:$chain, TypeAttr:$type);
  let results = (outs AnyTypeOf<[I1, I32]>:$result);

  let assemblyFormat = [{
    $buffer`,` $chain attr-dict `:` custom<TypeAttr>($type, type($result))
  }];
}

def TFRTGPU_MemPrintOp : TFRTGPU_Op<"mem.print_metadata"> {
  let description = [{
    tfrt_gpu.mem.print_metadata prints a gpu buffer metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.mem.print_metadata %buffer %ch1
  }];
  let arguments = (ins TFRTGPU_BufferType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

class TensorMakeOp<string dtype> : TFRTGPU_Op<"tensor.make." # dtype> {
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
  let arguments = (ins TFRTGPU_BufferType, TS_Shape, TFRT_ChainType);
  let results = (outs TensorType);
}

// TODO(csigg): this should just be a type attribute.
foreach dtype = ["i8", "i32", "i64", "f32", "f64"] in {
  def TFRTGPU_TensorMakeOp_#dtype : TensorMakeOp<dtype>;
}

def TFRTGPU_TensorPrintOp : TFRTGPU_Op<"tensor.print_metadata"> {
  let description = [{
    tfrt_gpu.tensor.print prints a gpu tensor metadata

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = tfrt_gpu.tensor.print_metadata %tensor, %ch1
  }];
  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def TFRTGPU_ModuleLoadOp : TFRTGPU_Op<"module.load"> {
  let description = [{
    tfrt_gpu.module.load loads GPU device module from a binary blob.

    The data attribute must be explicitly null-terminated.

    Example:
      %module = tfrt_gpu.module.load %ctx {
        data = "null-terminated device function blob, e.g. PTX"
      }
  }];
  let arguments = (ins TFRTGPU_ContextType:$ctx, StrAttr:$data);
  let results = (outs TFRTGPU_ModuleType);
}

def TFRTGPU_ModuleGetGlobalOp : TFRTGPU_Op<"module.get_global"> {
  let description = [{
    tfrt_gpu.module.get_global retrieves a GPU device global from a module.

    Example:
      %global = tfrt_gpu.module.get_global %module { name = "symbol" }
  }];
  let arguments = (ins TFRTGPU_ModuleType:$module, StrAttr:$name);
  let results = (outs TFRTGPU_BufferType);
}

def TFRTGPU_ModuleGetFunctionOp : TFRTGPU_Op<"module.get_function", [Pure]> {
  let description = [{
    tfrt_gpu.module.get_function retrieves a GPU device function from a module.

    Example:
      %function = tfrt_gpu.module.get_function %module { name = "kernel" }
  }];
  let arguments = (ins TFRTGPU_ModuleType:$module, StrAttr:$name);
  let results = (outs TFRTGPU_FunctionType);
}

def TFRTGPU_FunctionLaunchOp : TFRTGPU_Op<"function.launch"> {
  let description = [{
    tfrt_gpu.function.launch invokes the provided device function on a stream.

    Example:
      %ch1 = tfrt_gpu.function.launch %stream, %function,
                 blocks in (%grid_dim_x, %grid_dim_y, %grid_dim_z),
                 threads in (%block_dim_x, %block_dim_y, %block_dim_z),
                 %shared_memory_size_bytes, %chain,
                 args(%arg0, %arg1, ..., %argN) : (<type0>, <type1>, ... <typeN>)
  }];
  let arguments = (ins
    TFRTGPU_StreamType:$stream,
    TFRTGPU_FunctionType:$function,
    UI64:$grid_dim_x,
    UI64:$grid_dim_y,
    UI64:$grid_dim_z,
    UI64:$block_dim_x,
    UI64:$block_dim_y,
    UI64:$block_dim_z,
    UI32:$shared_memory_size_bytes,
    TFRT_ChainType:$in_chain,
    Variadic<AnyTypeOf<[TFRTGPU_BufferType, AnyFloat, AnyInteger]>>:$args
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

def TFRTGPU_AliasOp : TFRTGPU_Op<"alias", [
      Pure,
      TypesMatchWith<"value type matches result type", "res", "value", "$_self">
]>{

  let description = [{
    tfrt_gpu.alias returns the value argument once the input chain is ready.

    Example:
      %1 = tfrt_gpu.alias %0, %ch : !tfrt_gpu.module
  }];

  let arguments = (ins AnyType:$value, TFRT_ChainType:$chain);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    $value`,` $chain attr-dict `:` type($res)
  }];

}

#endif  // TFRTGPU_OPS
