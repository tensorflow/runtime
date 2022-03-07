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

// CHECK-LABEL: --- Running 'memcpy_host_to_device_and_back_test'
func @memcpy_host_to_device_and_back_test() {
  %ch0 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  %size = tfrt.constant.i64 32
  %gpu_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0

  // Create source dense host tensor.
  %src_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [8 : i64]
  %src_buffer, %ch12 = tfrt_dht.get_buffer %src_tensor, %ch0
  %ch1 = tfrt_dht.fill_tensor_with_constant.i32 %src_tensor, %ch0 1 : i32

  // CHECK: shape = [8], values = [1, 1, 1, 1, 1, 1, 1, 1]
  %ch2 = tfrt_dht.print_tensor %src_tensor, %ch1
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch3 = tfrt_dht.print_buffer %src_buffer, %ch2

  // Copy host to device.
  %ch4 = tfrt_gpu.mem.copy %gpu_buffer, %src_buffer, %stream, %ch3 : !tfrt_gpu.buffer, !ht.host_buffer

  // Create resulting dense host tensor, get its buffer, and copy back to host.
  %dst_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64, 4 : i64]
  %dst_buffer, %ch30 = tfrt_dht.get_buffer %dst_tensor, %ch0
  %ch5 = tfrt_gpu.mem.copy %dst_buffer, %gpu_buffer, %stream, %ch4 : !ht.host_buffer, !tfrt_gpu.buffer

  %ch6 = tfrt_gpu.stream.synchronize %stream, %ch5

  // CHECK: shape = [2, 4], values = [1, 1, 1, 1, 1, 1, 1, 1]
  %ch7 = tfrt_dht.print_tensor %dst_tensor, %ch6
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch8 = tfrt_dht.print_buffer %dst_buffer, %ch7

  tfrt.return
}
