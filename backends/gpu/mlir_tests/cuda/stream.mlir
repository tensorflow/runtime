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

// CHECK-LABEL: --- Running 'stream_create_test'
func @stream_create_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %allocator = tfrt_gpu.allocator.create %context, %ch2
  %stream = tfrt_gpu.stream.create %context, %ch2

  %size = tfrt.constant.i64 64
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch2
  // CHECK: GpuBuffer<pointer={{0x[[:xdigit:]]*}} (CUDA), size=64>
  %ch8 = tfrt_gpu.mem.print_metadata %buffer, %ch2

  %shape = ts.build_shape [2 : i64, 4 : i64]
  %tensor = tfrt_gpu.tensor.make.f64 %buffer, %shape, %ch8
  // CHECK: DenseGpuTensor<dtype=F64, shape=[2, 4], pointer={{0x[[:xdigit:]]*}} (CUDA)>
  %ch10 = tfrt_gpu.tensor.print_metadata %tensor, %ch8

  tfrt.return
}

// CHECK-LABEL: --- Running 'stream_create_synchronize'
func @stream_create_synchronize() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2

  %stream = tfrt_gpu.stream.create %context, %ch2
  %ch7 = tfrt_gpu.stream.synchronize %stream, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'make_tensor_from_smaller_buffer_should_fail'
func @make_tensor_from_smaller_buffer_should_fail() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %allocator = tfrt_gpu.allocator.create %context, %ch2
  %stream = tfrt_gpu.stream.create %context, %ch2

  %size = tfrt.constant.i64 64
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch2

  %shape = ts.build_shape [5 : i64, 4 : i64]
  // expected-error @+1 {{tfrt_gpu.tensor.make failed: buffer_size (64) is not equal to the number of elements in shape ([5, 4]) times element size (4)}}
  %tensor = tfrt_gpu.tensor.make.i32 %buffer, %shape, %ch2

  tfrt.return
}

