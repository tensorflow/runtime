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

// CHECK-LABEL: --- Running 'memset_test'
func @memset_test() {
  %ch1 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  // Initialize an empty device buffer
  %size = tfrt.constant.i64 32
  %device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch1

  // Set device buffer to value.
  %value = tfrt.constant.i32 13
  %ch2 = tfrt_gpu.mem.set %device_buffer, %value, %stream, %ch1 : !tfrt_gpu.buffer, i32

  // Create a host tensor and copy the device buffer to host to check the value.
  %host_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [8 : i64]
  %host_buffer, %ch3 = tfrt_dht.get_buffer %host_tensor, %ch2
  %ch4 = tfrt_gpu.mem.copy %host_buffer, %device_buffer, %stream, %ch3 : !ht.host_buffer, !tfrt_gpu.buffer

  // CHECK: shape = [8], values = [13, 13, 13, 13, 13, 13, 13, 13]
  %ch5 = tfrt_dht.print_tensor %host_tensor, %ch4
  tfrt.return
}
