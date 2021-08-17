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

// RUN: bef_executor %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'mem_register'
func @mem_register() {
  %ch0 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %stream = tfrt_gpu.stream.create %context

  %host_tensor = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %host_buffer:2 = tfrt_dht.get_buffer %host_tensor, %ch0
  %register_buffer = tfrt_gpu.mem.register %context, %host_buffer#0

  %module = tfrt_gpu.module.load %context {
    // __global__ void Kernel(int* ptr) { *ptr = 42; }
    data = ".version 6.0\n.target sm_60\n.address_size 64\n.visible .entry Kernel(.param .u64 ptr) {\n.reg .b32 %r<2>;\n.reg .b64 %rd<3>;\nld.param.u64 %rd1, [ptr];\ncvta.to.global.u64 %rd2, %rd1;\nmov.u32 %r1, 42;\nst.global.u32 [%rd2], %r1;\nret;\n}\00",
    key = 0 : ui64
  }
  %func = tfrt_gpu.function.get %module { name = "Kernel" }

  %zero = tfrt.constant.ui32 0
  %one = tfrt.constant.ui32 1
  %ch1 = tfrt_gpu.function.launch %stream, %func,
             blocks in (%one, %one, %one),
             threads in (%one, %one, %one),
             %zero, %ch0,
             args(%register_buffer) : (!tfrt_gpu.buffer)

  %ch2 = tfrt_gpu.stream.synchronize %stream, %ch1

  // CHECK: DenseHostTensor dtype = i32, shape = [], values = [42]
  %ch3 = tfrt_dht.print_tensor %host_tensor, %ch2

  tfrt.return
}
