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


// CHECK-LABEL: --- Running 'cublas_axpy'
func @cublas_axpy() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = cuda.init %ch0
  %index = tfrt.constant.i32 0
  %device, %ch2 = cuda.device.get %index, %ch1
  %context, %ch3 = cuda_test.context.get %device, %ch2
  %allocator, %ch4 = cuda.allocator.create %context, %ch3
  %stream, %ch5 = cuda.stream.create %context, %ch4
  %cublas_handle = cuda.blas.create %context

  %cha0 = tfrt.merge.chains %ch5

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %tensor_0 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %cha1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_0, %cha0 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0, %cha2 = cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_0, %cha1


  %tensor_1 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %cha3 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_1, %cha0 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1, %cha4 = cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_1, %cha3

  %chb0 = tfrt.merge.chains %cha2, %cha4

  %one = tfrt.constant.i32 1
  %alpha = tfrt.constant.f32 1.0
  %chb1 = cuda.blas.axpy.f32 %context, %cublas_handle, %buffer_length, %alpha, %gpu_buffer_0, %one, %gpu_buffer_1, %one, %chb0

  %chc0 = tfrt.merge.chains %chb1

  // Copy result back
  %host_returned_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %host_returned_buffer, %chc1 = tfrt_dht.get_buffer %host_returned_tensor, %chc0

  %chc2 = cuda.mem.copy_device_to_host %context, %host_returned_buffer, %gpu_buffer_1, %buffer_size_in_bytes, %stream, %chc1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00]
  %chc3 = tfrt_dht.print_tensor %host_returned_tensor, %chc2

  %chd0 = tfrt.merge.chains %chc3

  %chd1 = cuda.allocator.destroy %allocator, %chd0

  tfrt.return %chd1 : !tfrt.chain
}


