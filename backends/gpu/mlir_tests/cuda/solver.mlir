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

// CHECK-LABEL: --- Running 'solver_potrf'
func @solver_potrf() {
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %solver = tfrt_gpu.solver.create %stream

  %dim = tfrt.constant.i32 2
  %buffer_size = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %ch1 = tfrt.new.chain

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch1
    [9.0 : f32, 6.0 : f32, 6.0 : f32, 5.0 : f32]

  %host_buffer, %ch3 = tfrt_dht.get_buffer %host_tensor, %ch1
  %gpu_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size, %ch1
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer, %host_buffer, %stream, %ch2 : !tfrt_gpu.buffer, !ht.host_buffer

  %workspace_size = tfrt_gpu.solver.potrf.buffer_size %solver,
    CUBLAS_FILL_MODE_LOWER, %dim, CUDA_R_32F, %dim, %ch1
  %workspace = tfrt_gpu.mem.allocate %allocator, %stream, %workspace_size, %ch1

  %devinfo_size = tfrt.constant.i64 4  // 4 bytes int
  %devinfo = tfrt_gpu.mem.allocate %allocator, %stream, %devinfo_size, %ch1

  %ch5 = tfrt_gpu.solver.potrf %solver, CUBLAS_FILL_MODE_LOWER, %dim,
    CUDA_R_32F, %gpu_buffer, %dim, %workspace, %devinfo, %ch4

  %ch6 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer, %stream, %ch5 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch7 = tfrt_gpu.stream.synchronize %stream, %ch6
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2], values =
  // CHECK-SAME: [3.000000e+00, 2.000000e+00, 6.000000e+00, 1.000000e+00]
  %ch8 = tfrt_dht.print_tensor %host_tensor, %ch7

  tfrt.return
}

// CHECK-LABEL: --- Running 'solver_potrf_batched'
func @solver_potrf_batched() {
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context
  %solver = tfrt_gpu.solver.create %stream

  %dim = tfrt.constant.i32 2
  %buffer_size = tfrt.constant.i64 16 // [2, 2] * 4 bytes floats = 16 bytes

  %ch1 = tfrt.new.chain

  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch1
    [9.0 : f32, 6.0 : f32, 6.0 : f32, 5.0 : f32]

  %host_buffer, %ch3 = tfrt_dht.get_buffer %host_tensor, %ch1
  %gpu_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size, %ch1
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer, %host_buffer, %stream, %ch2 : !tfrt_gpu.buffer, !ht.host_buffer

  %devinfo_size = tfrt.constant.i64 4  // 4 bytes int
  %devinfo = tfrt_gpu.mem.allocate %allocator, %stream, %devinfo_size, %ch1

  %batch_count = tfrt.constant.i32 1

  %ch5 = tfrt_gpu.solver.potrf.batch %solver, CUBLAS_FILL_MODE_LOWER, %dim,
    CUDA_R_32F, %gpu_buffer, %dim, %devinfo, %batch_count, %ch4

  %ch6 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer, %stream, %ch5 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch7 = tfrt_gpu.stream.synchronize %stream, %ch6
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2], values =
  // CHECK-SAME: [3.000000e+00, 2.000000e+00, 6.000000e+00, 1.000000e+00]
  %ch8 = tfrt_dht.print_tensor %host_tensor, %ch7

  tfrt.return
}
