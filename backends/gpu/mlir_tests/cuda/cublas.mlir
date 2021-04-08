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

func @create_two_tensors_on_gpu(%context: !tfrt_cuda.context, %allocator: !tfrt_cuda.allocator, %stream: !tfrt_cuda.stream, %chain: !tfrt.chain) -> (!tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain) {
  %tensor_0 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_0, %chain [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0, %ch2 = tfrt_cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_0, %ch1

  %tensor_1 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_1, %chain [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1, %ch4 = tfrt_cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_1, %ch3

  %ch_out = tfrt.merge.chains %ch2, %ch4

  tfrt.return %gpu_buffer_0, %gpu_buffer_1, %ch_out : !tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain
}


// CHECK-LABEL: --- Running 'cublas_axpy'
func @cublas_axpy() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_cuda.init %ch0
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch1
  %context, %ch3 = tfrt_cuda_test.context.get %device, %ch1
  %allocator, %ch4 = tfrt_cuda.allocator.create %context, %ch3
  %stream = tfrt_cuda.stream.create %context, %ch4
  %cublas_handle = tfrt_cuda.blas.create %context
  %ch6 = tfrt_cuda.blas.set_stream %cublas_handle, %stream, %ch4

  %cha0 = tfrt.merge.chains %ch6

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %gpu_buffer_0, %gpu_buffer_1, %cha1 = tfrt.call @create_two_tensors_on_gpu(%context, %allocator, %stream, %cha0) : (!tfrt_cuda.context, !tfrt_cuda.allocator, !tfrt_cuda.stream, !tfrt.chain) -> (!tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain)

  %chb0 = tfrt.merge.chains %cha1

  %one = tfrt.constant.i32 1
  %alpha = tfrt.constant.f32 1.0
  %chb1 = tfrt_cuda.blas.axpy.f32 %context, %cublas_handle, %buffer_length, %alpha, %gpu_buffer_0, %one, %gpu_buffer_1, %one, %chb0

  %chc0 = tfrt.merge.chains %chb1

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %chc1 = tfrt_dht.get_buffer %result_tensor, %chc0

  %chc2 = tfrt_cuda.mem.copy_device_to_host %context, %result_buffer, %gpu_buffer_1, %buffer_size_in_bytes, %stream, %chc1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00]
  %chc3 = tfrt_dht.print_tensor %result_tensor, %chc2

  %chd0 = tfrt.merge.chains %chc3

  %chd1 = tfrt_cuda.allocator.destroy %allocator, %chd0

  tfrt.return %chd1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'cublas_gemm'
func @cublas_gemm() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_cuda.init %ch0
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch1
  %context, %ch3 = tfrt_cuda_test.context.get %device, %ch1
  %allocator, %ch4 = tfrt_cuda.allocator.create %context, %ch3
  %stream = tfrt_cuda.stream.create %context, %ch4
  %cublas_handle = tfrt_cuda.blas.create %context
  %ch6 = tfrt_cuda.blas.set_stream %cublas_handle, %stream, %ch4

  %gpu_buffer_A, %gpu_buffer_B, %cha0 = tfrt.call @create_two_tensors_on_gpu(%context, %allocator, %stream, %ch6) : (!tfrt_cuda.context, !tfrt_cuda.allocator, !tfrt_cuda.stream, !tfrt.chain) -> (!tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %cha1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %cha0 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C, %cha2 = tfrt_cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_C, %cha1

  %dim = tfrt.constant.i32 2
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %cha3 = tfrt_cuda.blas.gemm.f32 %context, %cublas_handle, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %dim, %gpu_buffer_B, %dim, %beta, %gpu_buffer_C, %dim, %cha2 { transa = false, transb = false }

  %chb0 = tfrt.merge.chains %cha3

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %chb1 = tfrt_dht.get_buffer %result_tensor, %chb0

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %chb2 = tfrt_cuda.mem.copy_device_to_host %context, %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %chb1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %chb3 = tfrt_dht.print_tensor %result_tensor, %chb2

  %chc0 = tfrt.merge.chains %chb3
  %chc1 = tfrt_cuda.allocator.destroy %allocator, %chc0

  tfrt.return %chc1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'cublas_gemm_ex'
func @cublas_gemm_ex() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_cuda.init %ch0
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch1
  %context, %ch3 = tfrt_cuda_test.context.get %device, %ch1
  %allocator, %ch4 = tfrt_cuda.allocator.create %context, %ch3
  %stream = tfrt_cuda.stream.create %context, %ch4
  %cublas_handle = tfrt_cuda.blas.create %context
  %ch6 = tfrt_cuda.blas.set_stream %cublas_handle, %stream, %ch4

  %gpu_buffer_A, %gpu_buffer_B, %cha0 = tfrt.call @create_two_tensors_on_gpu(%context, %allocator, %stream, %ch6) : (!tfrt_cuda.context, !tfrt_cuda.allocator, !tfrt_cuda.stream, !tfrt.chain) -> (!tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %cha1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %cha0 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C, %cha2 = tfrt_cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_C, %cha1

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt.constant.i32 0
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %cha3 = tfrt_cuda.blas.gemm.ex %context, %cublas_handle, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %type, %dim, %gpu_buffer_B, %type, %dim, %beta, %gpu_buffer_C, %type, %dim, %type, %algo, %cha2 { transa = false, transb = false }

  %chb0 = tfrt.merge.chains %cha3

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %chb1 = tfrt_dht.get_buffer %result_tensor, %chb0

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %chb2 = tfrt_cuda.mem.copy_device_to_host %context, %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %chb1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %chb3 = tfrt_dht.print_tensor %result_tensor, %chb2

  %chc0 = tfrt.merge.chains %chb3
  %chc1 = tfrt_cuda.allocator.destroy %allocator, %chc0

  tfrt.return %chc1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'cublas_gemm_strided_batched_ex'
func @cublas_gemm_strided_batched_ex() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_cuda.init %ch0
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch1
  %context, %ch3 = tfrt_cuda_test.context.get %device, %ch1
  %allocator, %ch4 = tfrt_cuda.allocator.create %context, %ch3
  %stream = tfrt_cuda.stream.create %context, %ch4
  %cublas_handle = tfrt_cuda.blas.create %context
  %ch6 = tfrt_cuda.blas.set_stream %cublas_handle, %stream, %ch4

  %gpu_buffer_A, %gpu_buffer_B, %cha0 = tfrt.call @create_two_tensors_on_gpu(%context, %allocator, %stream, %ch6) : (!tfrt_cuda.context, !tfrt_cuda.allocator, !tfrt_cuda.stream, !tfrt.chain) -> (!tfrt_cuda.buffer, !tfrt_cuda.buffer, !tfrt.chain)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %cha1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %cha0 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C, %cha2 = tfrt_cuda_test.copy_tensor_host_to_device %context, %allocator, %stream, %tensor_C, %cha1

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt.constant.i32 0
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %batch_count = tfrt.constant.i32 1
  %stride = tfrt.constant.i64 1
  %cha3 = tfrt_cuda.blas.gemm.strided.batched.ex %context, %cublas_handle, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %type, %dim, %stride, %gpu_buffer_B, %type, %dim, %stride, %beta, %gpu_buffer_C, %type, %dim, %stride, %batch_count, %type, %algo, %cha2 { transa = false, transb = false }

  %chb0 = tfrt.merge.chains %cha3

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %chb1 = tfrt_dht.get_buffer %result_tensor, %chb0

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %chb2 = tfrt_cuda.mem.copy_device_to_host %context, %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %chb1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %chb3 = tfrt_dht.print_tensor %result_tensor, %chb2

  %chc0 = tfrt.merge.chains %chb3
  %chc1 = tfrt_cuda.allocator.destroy %allocator, %chc0

  tfrt.return %chc1 : !tfrt.chain
}


