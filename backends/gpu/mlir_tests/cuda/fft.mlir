// Copyright 2022 The TensorFlow Runtime Authors
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

// CHECK-LABEL: --- Running 'fft_test'
func @fft_test() {
  %ch1 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  // Set up input tensor.
  %host_tensor = tfrt_dht.create_uninitialized_tensor.f32.3 [2 : i64, 2 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.f32 %host_tensor, %ch1 [
    1.0 : f32,  0.0 : f32,  0.0 : f32, 1.0 : f32,
    0.0 : f32, -1.0 : f32, -1.0 : f32, 0.0 : f32
  ]

  // Copy input to GPU.
  %buffer_size_bytes = tfrt.constant.i64 32 // [2, 2, 2] * 4 bytes = 32 bytes
  %host_buffer, %ch3 = tfrt_dht.get_buffer %host_tensor, %ch1
  %gpu_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %buffer_size_bytes, %ch1
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer, %host_buffer, %stream, %ch2
      : !tfrt_gpu.buffer, !ht.host_buffer

  // Build and run fft
  %fft = tfrt_gpu.fft.create %context, CUFFT_C2C, 1, [2, 2], [4, 2, 1], [4, 2, 1]
  %workspace_size = tfrt_gpu.fft.get_workspace_size %fft
  %workspace = tfrt_gpu.mem.allocate %allocator, %stream, %workspace_size, %ch1
  %ch5 = tfrt_gpu.fft.execute
      %stream, %fft, %gpu_buffer, %gpu_buffer, %workspace, CUFFT_FORWARD, %ch4

  // Copy output to host.
  %ch6 = tfrt_gpu.mem.copy %host_buffer, %gpu_buffer, %stream, %ch5
      : !ht.host_buffer, !tfrt_gpu.buffer
  %ch7 = tfrt_gpu.stream.synchronize %stream, %ch6

  // Verify output.
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2, 2], values = [
  // CHECK-SAME: 0.000000e+00, 0.000000e+00, 2.000000e+00, -2.000000e+00,
  // CHECK-SAME: 2.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00]
  %ch8 = tfrt_dht.print_tensor %host_tensor, %ch7

  tfrt.return
}
