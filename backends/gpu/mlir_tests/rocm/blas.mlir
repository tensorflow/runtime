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

// RUN: bef_executor_lite %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'blas_axpy'
func @blas_axpy() {
  %ch1 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %context

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer_0, %host_buffer, %stream, %ch3 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy %gpu_buffer_1, %host_buffer, %stream, %ch5 : !tfrt_gpu.buffer, !ht.host_buffer

  %stride = tfrt.constant.i32 1
  %alpha = tfrt.constant.f32 1.0
  %ch7 = tfrt_gpu.blas.axpy %blas, %stream, %buffer_length, %alpha, rocblas_datatype_f32_r, 
         %gpu_buffer_0, rocblas_datatype_f32_r, %stride, %gpu_buffer_1, rocblas_datatype_f32_r, %stride,
         rocblas_datatype_f32_r, %ch6

  %ch8 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer_1, %stream, %ch7 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch9 = tfrt_gpu.stream.synchronize %stream, %ch8
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: values = [3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00]
  %ch10 = tfrt_dht.print_tensor %host_tensor, %ch9

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm'
func @blas_gemm() {
  %ch1 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %context

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer_0, %host_buffer, %stream, %ch3 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy %gpu_buffer_1, %host_buffer, %stream, %ch5 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch7 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch6 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_2 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch7
  %ch8 = tfrt_gpu.mem.copy %gpu_buffer_2, %host_buffer, %stream, %ch7 : !tfrt_gpu.buffer, !ht.host_buffer

  %dim = tfrt.constant.i32 2
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %algo = tfrt_gpu.blas.gemm.algo rocblas_gemm_algo_standard
  %ch9 = tfrt_gpu.blas.gemm %blas, %stream,
    rocblas_operation_none, rocblas_operation_none, %dim, %dim, %dim,
    %alpha, %gpu_buffer_0, rocblas_datatype_f32_r, %dim,
    %gpu_buffer_1, rocblas_datatype_f32_r, %dim, %beta,
    %gpu_buffer_2, rocblas_datatype_f32_r, %dim,
    rocblas_datatype_f32_r, %algo, %ch8

  %ch10 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer_2, %stream, %ch9 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch11 = tfrt_gpu.stream.synchronize %stream, %ch10
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch12 = tfrt_dht.print_tensor %host_tensor, %ch11

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_gemm_batched'
func @blas_gemm_batched() {
  %ch1 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %context

  %buffer_length = tfrt.constant.i32 4 // [2, 2] = 4 floats
  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch2 = tfrt_dht.get_buffer %host_tensor, %ch1

  %ch3 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch2 [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch3
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer_0, %host_buffer, %stream, %ch3 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch5 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch4 [2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch5
  %ch6 = tfrt_gpu.mem.copy %gpu_buffer_1, %host_buffer, %stream, %ch5 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch7 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch6 [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_2 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch7
  %ch8 = tfrt_gpu.mem.copy %gpu_buffer_2, %host_buffer, %stream, %ch7 : !tfrt_gpu.buffer, !ht.host_buffer

  %dim = tfrt.constant.i32 2
  %type = tfrt.constant.i32 0
  %algo = tfrt_gpu.blas.gemm.algo rocblas_gemm_algo_standard
  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 1.0
  %batch_count = tfrt.constant.i32 1
  %stride = tfrt.constant.i64 1
  %ch9 = tfrt_gpu.blas.gemm.batch %blas, %stream,
    rocblas_operation_none, rocblas_operation_none, %dim, %dim, %dim,
    %alpha, %gpu_buffer_0, rocblas_datatype_f32_r, %dim, %stride,
    %gpu_buffer_1, rocblas_datatype_f32_r, %dim, %stride, %beta,
    %gpu_buffer_2, rocblas_datatype_f32_r, %dim, %stride, %batch_count,
    rocblas_datatype_f32_r, %algo, %ch8

  %ch10 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer_2, %stream, %ch9 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch11 = tfrt_gpu.stream.synchronize %stream, %ch10
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: values = [1.100000e+01, 1.600000e+01, 1.900000e+01, 2.800000e+01]
  %ch12 = tfrt_dht.print_tensor %host_tensor, %ch11

  tfrt.return
}

// CHECK-LABEL: --- Running 'blas_trsm_batched'
func @blas_trsm_batched() {
  %ch0 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %blas = tfrt_gpu.blas.create %context

  %buffer_size_bytes = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %host_buffer, %ch1 = tfrt_dht.get_buffer %host_tensor, %ch0

  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch1 [1.0 : f32, 2.0 : f32, 0.0 : f32, 1.0 : f32]
  %gpu_buffer_0 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch2
  %ch3 = tfrt_gpu.mem.copy %gpu_buffer_0, %host_buffer, %stream, %ch2 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch4 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch3 [1.0 : f32, 4.0 : f32, 0.0 : f32, 0.0 : f32]
  %gpu_buffer_1 = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch4
  %ch5 = tfrt_gpu.mem.copy %gpu_buffer_1, %host_buffer, %stream, %ch4 : !tfrt_gpu.buffer, !ht.host_buffer

  %dim = tfrt.constant.i32 2
  %alpha = tfrt.constant.f32 1.0
  %batch_count = tfrt.constant.i32 1
  %ch6 = tfrt_gpu.blas.trsm.batch %blas, %stream, rocblas_side_left,
    rocblas_fill_lower, rocblas_operation_none, rocblas_diagonal_unit, %dim, %dim,
    rocblas_datatype_f32_r, %alpha, %gpu_buffer_0, %dim, %gpu_buffer_1, %dim, %batch_count,
    %ch5

  %ch7 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer_1, %stream, %ch6 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch8 = tfrt_gpu.stream.synchronize %stream, %ch7
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00]
  %ch9 = tfrt_dht.print_tensor %host_tensor, %ch8

  tfrt.return
}
