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

// RUN: bef_executor_lite %s.bef | FileCheck %s

// CHECK-LABEL: --- Not running 'cond' because it has arguments
func.func @cond(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> i1 {

  %context = tfrt_gpu.stream.get_context %stream
  %module = tfrt_gpu.module.load %context {
    // __global__ void Kernel(int* value, bool* cond) { *cond = *value < 8; }
    data = ".version 6.0\n.target sm_60\n.address_size 64\n\n.visible .entry Kernel(.param .u64 value, .param .u64 cond) {\n  .reg .pred %p<2>;\n  .reg .b16 %rs<2>;\n  .reg .b32 %r<2>;\n  .reg .b64 %rd<5>;\n  ld.param.u64 %rd1, [value];\n  ld.param.u64 %rd2, [cond];\n  cvta.to.global.u64 %rd3, %rd2;\n  cvta.to.global.u64 %rd4, %rd1;\n  ld.global.u32 %r1, [%rd4];\n  setp.lt.s32 %p1, %r1, 8;\n  selp.u16 %rs1, 1, 0, %p1;\n  st.global.u8 [%rd3], %rs1;\n  ret;\n}\00"
  }
  %func = tfrt_gpu.module.get_function %module { name = "Kernel" }

  %zero = tfrt.constant.ui32 0
  %one = tfrt.constant.ui64 1
  %ch1 = tfrt_gpu.function.launch %stream, %func,
             blocks in (%one, %one, %one),
             threads in (%one, %one, %one),
             %zero, %ch0,
             args(%value, %cond) : (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %ch2 = tfrt_gpu.stream.synchronize %stream, %ch1
  %result = tfrt_gpu.mem.load %cond, %ch2 : i1

  tfrt.return %result : i1
}

// Runs a kernel updating %value and %cond. Returns all arguments plus the value
// of %cond on the host.
// CHECK-LABEL: --- Not running 'body' because it has arguments
func.func @body(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer, i1) {
  %context = tfrt_gpu.stream.get_context %stream
  %module = tfrt_gpu.module.load %context {
    // __global__ void Kernel(int* value) { *value *= 2; }
    data = ".version 6.0\n.target sm_60\n.address_size 64\n\n.visible .entry Kernel(.param .u64 value) {\n  .reg .b32 %r<3>;\n  .reg .b64 %rd<3>;\n  ld.param.u64 %rd1, [value];\n  cvta.to.global.u64 %rd2, %rd1;\n  ld.global.u32 %r1, [%rd2];\n  shl.b32 %r2, %r1, 1;\n  st.global.u32 [%rd2], %r2;\n  ret;\n}\00"
  }
  %func = tfrt_gpu.module.get_function %module { name = "Kernel" }

  %zero = tfrt.constant.ui32 0
  %one = tfrt.constant.ui64 1
  %ch1 = tfrt_gpu.function.launch %stream, %func,
             blocks in (%one, %one, %one),
             threads in (%one, %one, %one),
             %zero, %ch0,
             args(%value) : (!tfrt_gpu.buffer)

  %result = tfrt.call @cond(%ch1, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer) -> i1

  tfrt.return %ch1, %stream, %value, %cond, %result
    : !tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer, i1
}

// Runs a kernel while %cond is true.
//
// It's a potential lowering of an lmhlo.while running on GPU.
// CHECK-LABEL: --- Not running 'while' because it has arguments
func.func @while(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> !tfrt.chain {
  %init = tfrt.call @cond(%ch0, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer) -> i1

  %result:4 = tfrt.while
    %init @body(%ch0, %stream, %value, %cond) parallel_iterations(1)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer)
    -> (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer)

  tfrt.return %result#0 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'main'
func.func @main() {
  %ch0 = tfrt.new.chain

  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  %value_size = tfrt.constant.i64 4
  %value_init = tfrt.constant.i32 1
  %value = tfrt_gpu.mem.allocate %allocator, %stream, %value_size, %ch0
  %ch1 = tfrt_gpu.mem.set %value, %value_init, %stream, %ch0 : i32

  %cond_size = tfrt.constant.i64 1
  %cond = tfrt_gpu.mem.allocate_host %context, %cond_size, %ch0

  %ch2 = tfrt.call @while(%ch1, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer) -> !tfrt.chain

  %result_ht = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %result_hb:2 = tfrt_dht.get_buffer %result_ht, %ch0
  %ch3 = tfrt_gpu.mem.copy %result_hb#0, %value, %stream, %ch2 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch4 = tfrt_gpu.stream.synchronize %stream, %ch3

  // CHECK: DenseHostTensor dtype = i32, shape = [], values = [8]
  %ch5 = tfrt_dht.print_tensor %result_ht, %ch4

  tfrt.return
}
