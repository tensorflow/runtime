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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: tfrt_gpu_opt %s | tfrt_gpu_opt

// CHECK-LABEL: --- Running 'noop_kernel'
func @noop_kernel() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %stream = tfrt_gpu.stream.create %context, %ch2

  %func = tfrt_gpu.function.load %context, %ch2 {
    // PTX for empty_kernel.
    data = ".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry empty_kernel() { ret; }\00",
    key = 0 : ui64,
    name = "empty_kernel\00"
  }

  %blk_dim = tfrt.constant.ui32 1
  %grid_dim = tfrt.constant.ui32 1
  %shared_mem_size = tfrt.constant.ui32 0

  %ch7 = tfrt_gpu.function.launch %stream, %func,
             blocks in (%grid_dim, %grid_dim, %grid_dim),
             threads in (%blk_dim, %blk_dim, %blk_dim),
             %shared_mem_size, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'vector_add_kernel'
func @vector_add_kernel() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %stream = tfrt_gpu.stream.create %context, %ch2
  %allocator = tfrt_gpu.allocator.create %context, %ch2

  %func = tfrt_gpu.function.load %context, %ch2 {
    // PTX for vector_add.
    data = ".version 6.4\n.target sm_30\n.address_size 64\n.visible .entry vector_add(\n.param .u32 vector_add_param_0,\n.param .u64 vector_add_param_1,\n.param .u64 vector_add_param_2\n)\n{\n.reg .pred 	%p<2>;\n.reg .f32 	%f<4>;\n.reg .b32 	%r<6>;\n.reg .b64 	%rd<8>;\nld.param.u32 	%r2, [vector_add_param_0];\nld.param.u64 	%rd1, [vector_add_param_1];\nld.param.u64 	%rd2, [vector_add_param_2];\nmov.u32 	%r3, %ctaid.x;\nmov.u32 	%r4, %ntid.x;\nmov.u32 	%r5, %tid.x;\nmad.lo.s32 	%r1, %r4, %r3, %r5;\nsetp.ge.s32	%p1, %r1, %r2;\n@%p1 bra 	BB0_2;\n\ncvta.to.global.u64 	%rd3, %rd2;\ncvta.to.global.u64 	%rd4, %rd1;\nmul.wide.s32 	%rd5, %r1, 4;\nadd.s64 	%rd6, %rd4, %rd5;\nadd.s64 	%rd7, %rd3, %rd5;\nld.global.f32 	%f1, [%rd7];\nld.global.f32 	%f2, [%rd6];\nadd.f32 	%f3, %f2, %f1;\nst.global.f32 	[%rd7], %f3;\nBB0_2:\nret;\n}\n\00",
    key = 1 : ui64,
    name = "vector_add\00"
  }

  // Create source dense host tensors.
  %x_host = tfrt_dht.create_uninitialized_tensor.f32.1 [8 : i64]
  %ch7 = tfrt_dht.fill_tensor_with_constant.f32 %x_host, %ch2 1.0 : f32
  // CHECK: shape = [8], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch8 = tfrt_dht.print_tensor %x_host, %ch7

  %y_host = tfrt_dht.create_uninitialized_tensor.f32.1 [8 : i64]
  %ch9 = tfrt_dht.fill_tensor_with_constant.f32 %y_host, %ch8 1.0 : f32
  // CHECK: shape = [8], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch10 = tfrt_dht.print_tensor %y_host, %ch9


  // Extract buffers.
  %x_host_buffer, %ch11 = tfrt_dht.get_buffer %x_host, %ch10
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch12 = tfrt_dht.print_buffer %x_host_buffer, %ch11
  %y_host_buffer, %ch13 = tfrt_dht.get_buffer %y_host, %ch12
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch14 = tfrt_dht.print_buffer %y_host_buffer, %ch13

  %size = tfrt.constant.i64 32
  %x_device = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch14
  %y_device = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch14

  // Copy host to device.
  %ch17 = tfrt_gpu.mem.copy_host_to_device %x_device, %x_host_buffer, %size, %stream, %ch14
  %ch18 = tfrt_gpu.mem.copy_host_to_device %y_device, %y_host_buffer, %size, %stream, %ch14

  %one = tfrt.constant.ui32 1
  %eight = tfrt.constant.ui32 8
  %shared_mem_size = tfrt.constant.ui32 0
  %len = tfrt.constant.i32 8

  %ch_kernel = tfrt_gpu.function.launch %stream, %func,
                   blocks in (%eight, %one, %one),
                   threads in (%eight, %one, %one),
                   %shared_mem_size, %ch18,
                   args(%len, %x_device, %y_device) : (i32, !tfrt_gpu.buffer, !tfrt_gpu.buffer)

  // Copy back to host buffer and synchronize.
  %ch19 = tfrt_gpu.mem.copy_device_to_host %y_host_buffer, %y_device, %size, %stream, %ch_kernel
  %sync_ch = tfrt_gpu.stream.synchronize %stream, %ch19

  // CHECK: shape = [8], values = [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
  %ch25 = tfrt_dht.print_tensor %y_host, %sync_ch

  tfrt.return
}
