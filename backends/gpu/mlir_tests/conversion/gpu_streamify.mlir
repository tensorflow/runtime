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

// RUN: tfrt_gpu_opt %s -gpu-tfrt-streamify | FileCheck %s

// CHECK-LABEL: @memset
func @memset(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream,
  %arg2 : !tfrt_gpu.buffer,
  %arg3 : i32
) {
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
    : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %dst = builtin.unrealized_conversion_cast %arg2
    : !tfrt_gpu.buffer to memref<32xi32>
  // CHECK: %[[ch:.*]] = tfrt_gpu.mem.set %arg2, %arg3, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, i32
  %t1 = gpu.memset async [%t0] %dst, %arg3 : memref<32xi32>, i32
  // CHECK: builtin.unrealized_conversion_cast %[[ch]], %arg1
  // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  tfrt.return
}

// CHECK-LABEL: @memcopy
func @memcopy(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream,
  %arg2 : !tfrt_gpu.buffer,
  %arg3 : !tfrt_gpu.buffer
) {
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
    : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %src = builtin.unrealized_conversion_cast %arg2
    : !tfrt_gpu.buffer to memref<32xi32>
  %dst = builtin.unrealized_conversion_cast %arg3
    : !tfrt_gpu.buffer to memref<32xi32>
  // CHECK: %[[ch:.*]] = tfrt_gpu.mem.copy %arg3, %arg2, %arg1, %arg0
  // CHECK-SAME: : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  %t1 = gpu.memcpy async [%t0] %dst, %src : memref<32xi32>, memref<32xi32>
  // CHECK: builtin.unrealized_conversion_cast %[[ch]], %arg1
  // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token

  %empty = builtin.unrealized_conversion_cast %arg2
    : !tfrt_gpu.buffer to memref<0xi32>
  // CHECK-NOT: tfrt_gpu.mem.copy
  %t2 = gpu.memcpy async [%t0] %empty, %empty : memref<0xi32>, memref<0xi32>

  tfrt.return
}

// CHECK-LABEL: @gpu_container_module
module @gpu_container_module attributes {gpu.container_module} {

  // CHECK-LABEL: func @zero(%arg0: !tfrt_gpu.context) -> !tfrt_gpu.buffer {
  // CHECK:   %[[allocator:.*]] = tfrt_gpu.allocator.create %arg0
  // CHECK:   %[[stream:.*]] = tfrt_gpu.stream.create %arg0
  // CHECK:   %[[size:.*]] = tfrt.constant.i64 4
  // CHECK:   %[[ch0:.*]] = tfrt.new.chain
  // CHECK:   %[[buffer:.*]] = tfrt_gpu.mem.allocate %[[allocator]], %[[stream]], %[[size]], %[[ch0]]
  // CHECK:   %[[tensor:.*]] = tfrt_dht.create_uninitialized_tensor.i32.0 []
  // CHECK:   %[[ch1:.*]] = tfrt_dht.set_tensor_with_constant_values.i32 %[[tensor]], %[[ch0]] [0 : i32]
  // CHECK:   %[[host:.*]]:2 = tfrt_dht.get_buffer %[[tensor]], %[[ch1]]
  // CHECK:   %[[ch2:.*]] = tfrt_gpu.mem.copy %4, %[[host]]#0, %[[stream]], %[[ch1]] : !tfrt_gpu.buffer, !ht.host_buffer
  // CHECK:   %[[ch3:.*]] = tfrt_gpu.stream.synchronize %[[stream]], %[[ch2]]
  // CHECK:   %[[alias:.*]] = tfrt_gpu.alias %[[buffer]], %[[ch3]] : !tfrt_gpu.buffer
  // CHECK:   tfrt.return %[[alias]] : !tfrt_gpu.buffer
  // CHECK: }
  memref.global @zero : memref<i32> = dense<0>

  // CHECK-LABEL: func @pi(%arg0: !tfrt_gpu.context) -> !tfrt_gpu.buffer {
  // CHECK:   %[[module:.*]] = tfrt.once @gpu_module(%arg0) : (!tfrt_gpu.context) -> (!tfrt_gpu.module)
  // CHECK:   %[[global:.*]] = tfrt_gpu.module.get_global %[[module]] {name = "pi"}
  // CHECK:   tfrt.return %[[global]] : !tfrt_gpu.buffer
  // CHECK: }
  memref.global @pi : memref<f32> = dense<3.14159274> {gpu_module = @gpu_module}

  // CHECK-LABEL: func @gpu_module(%arg0: !tfrt_gpu.context)
  // CHECK-SAME: -> !tfrt_gpu.module {
  // CHECK:   %[[module:.*]] = tfrt_gpu.module.load %arg0 {data = "<cubin>\00"}
  // CHECK:   %[[ch0:.*]] = tfrt.new.chain
  // CHECK:   %[[stream:.*]] = tfrt_gpu.stream.create %arg0
  // CHECK:   %[[global:.*]] = tfrt_gpu.module.get_global %[[module]] {name = "pi"}
  // CHECK:   %[[tensor:.*]] = tfrt_dht.create_uninitialized_tensor.f32.0 []
  // CHECK:   %[[ch1:.*]] = tfrt_dht.set_tensor_with_constant_values.f32
  // CHECK-SAME: %[[tensor]], %[[ch0]] [3.14159274 : f32]
  // CHECK:   %[[buffer:.*]]:2 = tfrt_dht.get_buffer %[[tensor]], %[[ch1]]
  // CHECK:   %[[ch2:.*]] = tfrt_gpu.mem.copy %[[global]], %[[buffer]]#0,
  // CHECK-SAME: %[[stream]], %[[ch1]] : !tfrt_gpu.buffer, !ht.host_buffer
  // CHECK:   %[[ch3:.*]] = tfrt_gpu.stream.synchronize %[[stream]], %[[ch2]]
  // CHECK:   %[[alias:.*]] = tfrt_gpu.alias %[[module]], %[[ch3]] : !tfrt_gpu.module
  // CHECK:   tfrt.return %[[alias]] : !tfrt_gpu.module
  // CHECK: }
  gpu.module @gpu_module attributes {
    binary = "<cubin>",
    constants = [@pi]
  } {
    gpu.func @kernel() kernel { gpu.return }
  }

  // CHECK-LABEL: @get_global
  func @get_global(%arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.stream) {
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
    // CHECK: %[[zero:.*]] = tfrt.once @zero(%[[ctx]])
    // CHECK: builtin.unrealized_conversion_cast %[[zero]]
    // CHECK-SAME: : !tfrt_gpu.buffer to memref<i32>
    %zero = memref.get_global @zero : memref<i32>
    // CHECK: %[[pi:.*]] = tfrt.once @pi(%[[ctx]])
    // CHECK: builtin.unrealized_conversion_cast %[[pi]]
    // CHECK-SAME: : !tfrt_gpu.buffer to memref<f32>
    %pi = memref.get_global @pi : memref<f32>
    tfrt.return
  }

  // CHECK-LABEL: @launch
  func @launch(%arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.stream) {
    %one = arith.constant 1 : index
    %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
        : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
    // CHECK: %[[once:.*]] = tfrt.once @gpu_module(%[[ctx]])
    // CHECK-SAME: (!tfrt_gpu.context) -> (!tfrt_gpu.module)
    // CHECK: %[[kernel:.*]] = tfrt_gpu.module.get_function %[[once]] {name = "kernel"}
    // CHECK: %[[ch0:.*]] = tfrt_gpu.function.launch %arg1, %[[kernel]],
    // CHECK-SAME: blocks in ({{.*}}), threads in ({{.*}}), {{.*}}, %arg0
    %t1 = gpu.launch_func async [%t0] @gpu_module::@kernel
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
    tfrt.return
  }
}

// CHECK-LABEL: @unwrap_async
func @unwrap_async(%arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.stream) {
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t1 = "tfrt_gpu_conversion.async.execute"(%t0) ({
  ^bb0(%ch0: !tfrt.chain, %stream: !tfrt_gpu.stream):
    // CHECK: %[[ch:.*]] = tfrt_gpu.stream.synchronize %arg1, %arg0
    %ch1 = tfrt_gpu.stream.synchronize %stream, %ch0
    tfrt.return %ch1 : !tfrt.chain
  }) : (!gpu.async.token) -> (!gpu.async.token)
  // CHECK: builtin.unrealized_conversion_cast %[[ch]], %arg1
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  tfrt.return
}

// CHECK-LABEL: @gpu_wait_remove
func @gpu_wait_remove(%arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.stream) {
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  // CHECK-NOT: gpu.wait
  %t1 = gpu.wait async [%t0]
  tfrt.return
}

// CHECK-LABEL: @gpu_wait_new_stream
func @gpu_wait_new_stream(%arg0 : !tfrt.chain, %arg1 : !tfrt_gpu.event) {
  // CHECK: %[[s0:.*]] = builtin.unrealized_conversion_cast to !tfrt_gpu.stream
  %stream = builtin.unrealized_conversion_cast to !tfrt_gpu.stream
  // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %[[s0]]
  %ctx = tfrt_gpu.stream.get_context %stream
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.event to !gpu.async.token
  // CHECK: %[[s1:.*]] = tfrt_gpu.stream.create %[[ctx]]
  // CHECK: %[[ch:.*]] = tfrt_gpu.stream.wait %[[s1]], %arg1, %arg0
  // CHECK: %[[t1:.*]] = builtin.unrealized_conversion_cast %[[ch]], %[[s1]]
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t1 = gpu.wait async [%t0]
  tfrt.return
}

// CHECK-LABEL: @gpu_wait_synchronize
func @gpu_wait_synchronize(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream,
  %arg2 : !tfrt.chain,
  %arg3 : !tfrt_gpu.event
) {
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t1 = builtin.unrealized_conversion_cast %arg2, %arg3
      : !tfrt.chain, !tfrt_gpu.event to !gpu.async.token
  // CHECK: %[[ch0:.*]] = tfrt.merge.chains %arg0, %arg2 : !tfrt.chain, !tfrt.chain
  // CHECK: %[[ch1:.*]] = tfrt_gpu.stream.wait %arg1, %arg3, %[[ch0]]
  // CHECK: builtin.unrealized_conversion_cast %[[ch1]], %arg1
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t4 = gpu.wait async [%t0, %t1]
  tfrt.return
}

// CHECK-LABEL: @gpu_wait_new_event
func @gpu_wait_new_event(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> !gpu.async.token {
  // CHECK: %[[t0:.*]] = builtin.unrealized_conversion_cast %arg0, %arg1
  // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t0 = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  // CHECK: %[[event:.*]]:2 = builtin.unrealized_conversion_cast %0
  // CHECK-SAME: !gpu.async.token to !tfrt.chain, !tfrt_gpu.event
  // CHECK: %[[t1:.*]] = builtin.unrealized_conversion_cast %[[event]]#0, %arg1
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  // CHECK: %[[t2:.*]] = builtin.unrealized_conversion_cast %[[event]]#0, %[[event]]#1
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.event to !gpu.async.token
  %t1 = gpu.wait async [%t0]
  // CHECK: tfrt.return %[[t2]] : !gpu.async.token
  tfrt.return %t1 : !gpu.async.token
}
