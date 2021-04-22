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

func @create_two_tensors_on_gpu(%allocator: !tfrt_gpu.allocator, %stream: !tfrt_gpu.stream) -> (!tfrt_gpu.buffer, !tfrt_gpu.buffer) {
  %chain = tfrt.new.chain

  %tensor_0 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_0, %chain [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor_0, %ch1

  %tensor_1 = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_1, %chain [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor_1, %ch3

  tfrt.return %gpu_buffer_0, %gpu_buffer_1 : !tfrt_gpu.buffer, !tfrt_gpu.buffer
}


// CHECK-LABEL: --- Running 'blas_axpy'
func @blas_axpy() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch1
  %allocator = tfrt_gpu.allocator.create %context, %ch1
  %stream = tfrt_gpu.stream.create %context, %ch1
  %blas = tfrt_gpu.blas.create %stream, %ch1

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %gpu_buffer_0, %gpu_buffer_1 = tfrt.call @create_two_tensors_on_gpu(%allocator, %stream) : (!tfrt_gpu.allocator, !tfrt_gpu.stream) -> (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %one = tfrt.constant.i32 1
  %alpha = tfrt.constant.f32 1.0
  %ch2 = tfrt_gpu.blas.axpy.f32 %blas, %buffer_length, %alpha, %gpu_buffer_0, %one, %gpu_buffer_1, %one, %ch1

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %ch3 = tfrt_dht.get_buffer %result_tensor, %ch2

  %ch4 = tfrt_gpu.mem.copy_device_to_host %result_buffer, %gpu_buffer_1, %buffer_size_in_bytes, %stream, %ch3
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00]
  %ch5 = tfrt_dht.print_tensor %result_tensor, %ch4

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm'
func @blas_gemm() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch1
  %allocator = tfrt_gpu.allocator.create %context, %ch1
  %stream = tfrt_gpu.stream.create %context, %ch1
  %blas = tfrt_gpu.blas.create %stream, %ch1

  %gpu_buffer_A, %gpu_buffer_B = tfrt.call @create_two_tensors_on_gpu(%allocator, %stream) : (!tfrt_gpu.allocator, !tfrt_gpu.stream) -> (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %ch1 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor_C, %ch2

  %dim = tfrt.constant.i32 2
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %ch4 = tfrt_gpu.blas.gemm.f32 %blas, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %dim, %gpu_buffer_B, %dim, %beta, %gpu_buffer_C, %dim, %ch2 { transa = false, transb = false }

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %ch5 = tfrt_dht.get_buffer %result_tensor, %ch4

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %ch6 = tfrt_gpu.mem.copy_device_to_host %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %ch5
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch7 = tfrt_dht.print_tensor %result_tensor, %ch6

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm_ex'
func @blas_gemm_ex() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch1
  %allocator = tfrt_gpu.allocator.create %context, %ch1
  %stream = tfrt_gpu.stream.create %context, %ch1
  %blas = tfrt_gpu.blas.create %stream, %ch1

  %gpu_buffer_A, %gpu_buffer_B = tfrt.call @create_two_tensors_on_gpu(%allocator, %stream) : (!tfrt_gpu.allocator, !tfrt_gpu.stream) -> (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %ch1 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor_C, %ch2

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt.constant.i32 0
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %ch4 = tfrt_gpu.blas.gemm.ex %blas, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %type, %dim, %gpu_buffer_B, %type, %dim, %beta, %gpu_buffer_C, %type, %dim, %type, %algo, %ch2 { transa = false, transb = false }

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %ch5 = tfrt_dht.get_buffer %result_tensor, %ch4

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %ch6 = tfrt_gpu.mem.copy_device_to_host %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %ch5
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch7 = tfrt_dht.print_tensor %result_tensor, %ch6

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm_strided_batched_ex'
func @blas_gemm_strided_batched_ex() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch1 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch1
  %allocator = tfrt_gpu.allocator.create %context, %ch1
  %stream = tfrt_gpu.stream.create %context, %ch1
  %blas = tfrt_gpu.blas.create %stream, %ch1

  %gpu_buffer_A, %gpu_buffer_B = tfrt.call @create_two_tensors_on_gpu(%allocator, %stream) : (!tfrt_gpu.allocator, !tfrt_gpu.stream) -> (!tfrt_gpu.buffer, !tfrt_gpu.buffer)

  %tensor_C = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor_C, %ch1 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_C = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor_C, %ch2

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt.constant.i32 0
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %batch_count = tfrt.constant.i32 1
  %stride = tfrt.constant.i64 1
  %ch4 = tfrt_gpu.blas.gemm.strided.batched.ex %blas, %dim, %dim, %dim, %alpha, %gpu_buffer_A, %type, %dim, %stride, %gpu_buffer_B, %type, %dim, %stride, %beta, %gpu_buffer_C, %type, %dim, %stride, %batch_count, %type, %algo, %ch2 { transa = false, transb = false }

  // Copy result back
  %result_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2: i64, 2: i64]
  %result_buffer, %ch5 = tfrt_dht.get_buffer %result_tensor, %ch4

  %buffer_size_in_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes
  %ch6 = tfrt_gpu.mem.copy_device_to_host %result_buffer, %gpu_buffer_C, %buffer_size_in_bytes, %stream, %ch5
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch7 = tfrt_dht.print_tensor %result_tensor, %ch6

  tfrt.return
}


