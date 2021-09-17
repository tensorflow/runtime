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
func @cond(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> i1 {

  %context = tfrt_gpu.stream.get_context %stream
  %module = tfrt_gpu.module.load %context {
    // __global__ void Kernel(int* value, bool* cond) { *cond = *value < 8; }
    data = ".version 6.0\n.target sm_60\n.address_size 64\n\n.visible .entry Kernel(.param .u64 value, .param .u64 cond) {\n  .reg .pred %p<2>;\n  .reg .b16 %rs<2>;\n  .reg .b32 %r<2>;\n  .reg .b64 %rd<5>;\n  ld.param.u64 %rd1, [value];\n  ld.param.u64 %rd2, [cond];\n  cvta.to.global.u64 %rd3, %rd2;\n  cvta.to.global.u64 %rd4, %rd1;\n  ld.global.u32 %r1, [%rd4];\n  setp.lt.s32 %p1, %r1, 8;\n  selp.u16 %rs1, 1, 0, %p1;\n  st.global.u8 [%rd3], %rs1;\n  ret;\n}\00",
    key = 0 : ui64
  }
  %func = tfrt_gpu.module.get_function %module { name = "Kernel" }

  %zero = tfrt.constant.ui32 0
  %one = tfrt.constant.ui32 1
  %ch1 = tfrt_gpu.function.launch %stream, %func,
             blocks in (%one, %one, %one),
             threads in (%one, %one, %one),
             %zero, %ch0,
             args(%value, %cond) : (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %cond_ht = tfrt_dht.create_uninitialized_tensor.bool.0 []
  %cond_hb:2 = tfrt_dht.get_buffer %cond_ht, %ch0

  %ch2 = tfrt_gpu.mem.copy %cond_hb#0, %cond, %stream, %ch1 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch3 = tfrt_gpu.stream.synchronize %stream, %ch2

  // '%result = tfrt_dht.get_value %cond_ht', but that kernel does not exist.
  %true_ht = tfrt_dht.create_uninitialized_tensor.bool.0 []
  %ch4 = tfrt_dht.fill_tensor_with_constant.bool %true_ht, %ch3 1 : i1
  %result:2 = tfrt_dht.tensor_equal.bool %cond_ht, %true_ht, %ch4

  tfrt.return %result#0 : i1
}

// Runs a kernel updating %value and %cond. Returns all arguments plus the value
// of %cond on the host.
// CHECK-LABEL: --- Not running 'body' because it has arguments
func @body(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer, i1) {
  %context = tfrt_gpu.stream.get_context %stream
  %module = tfrt_gpu.module.load %context {
    // __global__ void Kernel(int* value) { *value *= 2; }
    data = ".version 6.0\n.target sm_60\n.address_size 64\n\n.visible .entry Kernel(.param .u64 value) {\n  .reg .b32 %r<3>;\n  .reg .b64 %rd<3>;\n  ld.param.u64 %rd1, [value];\n  cvta.to.global.u64 %rd2, %rd1;\n  ld.global.u32 %r1, [%rd2];\n  shl.b32 %r2, %r1, 1;\n  st.global.u32 [%rd2], %r2;\n  ret;\n}\00",
    key = 1 : ui64
  }
  %func = tfrt_gpu.module.get_function %module { name = "Kernel" }

  %zero = tfrt.constant.ui32 0
  %one = tfrt.constant.ui32 1
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
func @while(
  %ch0    : !tfrt.chain,
  %stream : !tfrt_gpu.stream,
  %value  : !tfrt_gpu.buffer,
  %cond   : !tfrt_gpu.buffer
) -> !tfrt.chain {
  %init = tfrt.call @cond(%ch0, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer) -> i1

  %result:4 = tfrt.while
    %init @body(%ch0, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer)
    -> (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer)

  tfrt.return %result#0 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'main'
func @main() {
  %ch0 = tfrt.new.chain

  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  %value_size = tfrt.constant.i64 4
  %value = tfrt_gpu.mem.allocate %allocator, %stream, %value_size, %ch0
  %value_ht = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %value_hb:2 = tfrt_dht.get_buffer %value_ht, %ch0
  %ch1 = tfrt_dht.fill_tensor_with_constant.i32 %value_ht, %ch0 1 : i32
  %ch2 = tfrt_gpu.mem.copy %value, %value_hb#0, %stream, %ch1 : !tfrt_gpu.buffer, !ht.host_buffer

  %cond_size = tfrt.constant.i64 1
  %cond = tfrt_gpu.mem.allocate %allocator, %stream, %cond_size, %ch0
  %cond_ht = tfrt_dht.create_uninitialized_tensor.bool.0 []
  %cond_hb:2 = tfrt_dht.get_buffer %cond_ht, %ch0
  %ch3 = tfrt_dht.fill_tensor_with_constant.bool %cond_ht, %ch2 1 : i1
  %ch4 = tfrt_gpu.mem.copy %cond, %cond_hb#0, %stream, %ch3 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch5 = tfrt.call @while(%ch4, %stream, %value, %cond)
    :  (!tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer, !tfrt_gpu.buffer) -> !tfrt.chain

  %ch6 = tfrt_gpu.mem.copy %value_hb#0, %value, %stream, %ch5 : !ht.host_buffer, !tfrt_gpu.buffer
  // CHECK: DenseHostTensor dtype = i32, shape = [], values = [8]
  %ch7 = tfrt_dht.print_tensor %value_ht, %ch6

  tfrt.return
}
