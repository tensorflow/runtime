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
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch5 = tfrt_cuda_test.context.get %device, %ch2
  %allocator, %ch_alloc = tfrt_cuda.allocator.create %context, %ch2
  %stream = tfrt_cuda.stream.create %context, %ch2

  %size = tfrt.constant.i64 64
  %buffer, %ch7 = tfrt_cuda.mem.allocate %allocator, %stream, %size, %ch2
  // CHECK: GpuBuffer<pointer={{0x[[:xdigit:]]*}} (CUDA), size=64>
  %ch8 = tfrt_cuda.mem.print_metadata %buffer, %ch2

  %shape = ts.build_shape [2 : i64, 4 : i64]
  %tensor, %ch9 = tfrt_cuda.tensor.make.f64 %buffer, %shape, %ch8
  // CHECK: DenseGpuTensor<dtype=F64, shape=[2, 4], pointer={{0x[[:xdigit:]]*}} (CUDA)>
  %ch10 = tfrt_cuda.tensor.print_metadata %tensor, %ch9

  %ch11 = tfrt_cuda.allocator.destroy %allocator, %ch10
  tfrt.return
}

// CHECK-LABEL: --- Running 'stream_create_synchronize'
func @stream_create_synchronize() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1

  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch5 = tfrt_cuda_test.context.get %device, %ch2

  %stream = tfrt_cuda.stream.create %context, %ch2
  %ch7 = tfrt_cuda.stream.synchronize %stream, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'make_tensor_from_smaller_buffer_should_fail'
func @make_tensor_from_smaller_buffer_should_fail() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch5 = tfrt_cuda_test.context.get %device, %ch2
  %allocator, %ch_alloc = tfrt_cuda.allocator.create %context, %ch2
  %stream = tfrt_cuda.stream.create %context, %ch2

  %size = tfrt.constant.i64 64
  %buffer, %ch7 = tfrt_cuda.mem.allocate %allocator, %stream, %size, %ch2

  %shape = ts.build_shape [5 : i64, 4 : i64]
  // expected-error @+1 {{tfrt_cuda.tensor.make failed: buffer_size (64) is not equal to the number of elements in shape ([5, 4]) times element size (4)}}
  %tensor, %ch8 = tfrt_cuda.tensor.make.i32 %buffer, %shape, %ch7

  %ch10 = tfrt_cuda.allocator.destroy %allocator, %ch8
  tfrt.return
}

