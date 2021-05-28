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

// CHECK-LABEL: --- Running 'blas_axpy'
func @blas_axpy() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %index
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %stream

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_0, %host_buffer, %buffer_size_bytes, %stream, %ch3

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_1, %host_buffer, %buffer_size_bytes, %stream, %ch5

  %stride = tfrt.constant.i32 1
  %alpha = tfrt.constant.f32 1.0
  %ch7 = tfrt_gpu.blas.axpy %blas, %buffer_length, %alpha, CUDA_R_32F,
    %gpu_buffer_0, CUDA_R_32F, %stride, %gpu_buffer_1, CUDA_R_32F, %stride,
    CUDA_R_32F, %ch6

  %ch8 = tfrt_gpu.mem.copy_device_to_host %host_buffer, %gpu_buffer_1, %buffer_size_bytes, %stream, %ch7
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00]
  %ch9 = tfrt_dht.print_tensor %host_tensor, %ch8

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm'
func @blas_gemm() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %index
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %stream

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_0, %host_buffer, %buffer_size_bytes, %stream, %ch3

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_1, %host_buffer, %buffer_size_bytes, %stream, %ch5

  %ch7 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch6 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_2 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch7
  %ch8 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_2, %host_buffer, %buffer_size_bytes, %stream, %ch7

  %dim = tfrt.constant.i32 2
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %algo = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_ALGO0
  %ch9 = tfrt_gpu.blas.gemm %blas,
    CUBLAS_OP_N, CUBLAS_OP_N, %dim, %dim, %dim,
    %alpha, %gpu_buffer_0, CUDA_R_32F, %dim,
    %gpu_buffer_1, CUDA_R_32F, %dim, %beta,
    %gpu_buffer_2, CUDA_R_32F, %dim,
    CUDA_R_32F, %algo, %ch8

  %ch10 = tfrt_gpu.mem.copy_device_to_host %host_buffer, %gpu_buffer_2, %buffer_size_bytes, %stream, %ch9
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch11 = tfrt_dht.print_tensor %host_tensor, %ch10

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm_batched'
func @blas_gemm_batched() {
  %ch1 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %index
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %stream

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_0, %host_buffer, %buffer_size_bytes, %stream, %ch3

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_1, %host_buffer, %buffer_size_bytes, %stream, %ch5

  %ch7 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch6 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_2 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch7
  %ch8 = tfrt_gpu.mem.copy_host_to_device %gpu_buffer_2, %host_buffer, %buffer_size_bytes, %stream, %ch7

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_DEFAULT
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %batch_count = tfrt.constant.i32 1
  %stride = tfrt.constant.i64 1
  %ch9 = tfrt_gpu.blas.gemm.batch %blas,
    CUBLAS_OP_N, CUBLAS_OP_N, %dim, %dim, %dim,
    %alpha, %gpu_buffer_0, CUDA_R_32F, %dim, %stride,
    %gpu_buffer_1, CUDA_R_32F, %dim, %stride, %beta,
    %gpu_buffer_2, CUDA_R_32F, %dim, %stride, %batch_count,
    CUDA_R_32F, %algo, %ch8

  %ch10 = tfrt_gpu.mem.copy_device_to_host %host_buffer, %gpu_buffer_2, %buffer_size_bytes, %stream, %ch9
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch11 = tfrt_dht.print_tensor %host_tensor, %ch10

  tfrt.return
}


