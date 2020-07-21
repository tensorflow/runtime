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
// RUN: tfrt_opt %s | tfrt_opt

// CHECK-LABEL: --- Running 'memcpy_host_to_device_and_back_test'
func @memcpy_host_to_device_and_back_test() {
  %ch1 = tfrt.new.chain
  %ch2 = cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device, %ch4 = cuda.device.get %index, %ch2
  %context, %ch5 = cuda_test.context.get %device, %ch2
  %allocator, %ch_alloc = cuda.allocator.create %context, %ch2
  %stream, %ch6 = cuda.stream.create %context, %ch2

  %size = tfrt.constant.i64 32
  %device_buffer, %ch7 = cuda.mem.allocate %allocator, %stream, %size, %ch2

  // Create source dense host tensor.
  %host_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [8 : i64]
  %ch10 = tfrt_dht.fill_tensor_with_constant.i32 %host_tensor, %ch6 1 : i32
  // CHECK: shape = [8], values = [1, 1, 1, 1, 1, 1, 1, 1]
  %ch11 = tfrt_dht.print_tensor %host_tensor, %ch10
  %host_buffer, %ch12 = tfrt_dht.get_buffer %host_tensor, %ch1
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch13 = tfrt_dht.print_buffer %host_buffer, %ch1

  // Copy host to device.
  %ch20 = cuda.mem.copy_host_to_device %context, %device_buffer, %host_buffer, %size, %stream, %ch10

  // Create resulting dense host tensor, get its buffer, and copy back to host.
  %result_host_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64, 4 : i64]
  %result_host_buffer, %ch30 = tfrt_dht.get_buffer %result_host_tensor, %ch1
  %ch31 = cuda.mem.copy_device_to_host %context, %result_host_buffer, %device_buffer, %size, %stream, %ch20

  // Create, record, and poll an event to make sure copy back to host completed.
  %event, %ch40 = cuda.event.create %context, %ch2
  %ch41 = cuda.event.record %event, %stream, %ch31
  %ch42 = cuda.event.poll %event, %ch41

  // CHECK: shape = [2, 4], values = [1, 1, 1, 1, 1, 1, 1, 1]
  %ch50 = tfrt_dht.print_tensor %result_host_tensor, %ch42

  // This print is just to make sure (in non error cases) that source host
  // buffer stays alive for the duration of the copy.
  // TODO(iga): Add EventMgr and extend the life of host buffer inside memcpy.
  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=32>
  %ch51 = tfrt_dht.print_buffer %host_buffer, %ch42

  %ch60 = cuda.allocator.destroy %allocator, %ch50
  tfrt.return
}
