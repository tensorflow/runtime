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

// RUN: tfrt_gpu_opt %s -gpu-to-tfrt-gpu | FileCheck %s

// CHECK-LABEL: test_unwrap_async(
// CHECK-SAME:    %arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream
// CHECK-SAME:  ) -> !tfrt.chain {
func @test_unwrap_async() {
  %t0 = gpu.wait async
  %t1 = "tfrt_gpu_conversion.async.execute"(%t0) ( {
  ^bb0(%ch0: !tfrt.chain, %stream: !tfrt_gpu.stream):
    // CHECK-NEXT: %[[ch:.*]] = tfrt.merge.chains %arg0, %arg1 : !tfrt.chain, !tfrt_gpu.stream
    %ch1 = tfrt.merge.chains %ch0, %stream : !tfrt.chain, !tfrt_gpu.stream
    tfrt.return %ch1 : !tfrt.chain
  }) : (!gpu.async.token) -> (!gpu.async.token)
  gpu.wait [%t1]
  // CHECK-NEXT: tfrt.return %[[ch]] : !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: @test_skip_signature_rewrite() {
func @test_skip_signature_rewrite() {
  tfrt.return
}

// CHECK-LABEL: @test_erase_gpu_wait(
// CHECK-SAME:    %arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream
// CHECK-SAME:  ) -> !tfrt.chain {
func @test_erase_gpu_wait() {
  // These two ops are folded away completely.
  %t0 = gpu.wait async
  gpu.wait [%t0]
  // CHECK-NEXT: tfrt.return %arg0 : !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: @test_async_execute(
// CHECK-SAME:    %arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream
// CHECK-SAME:  ) -> !tfrt.chain {
func @test_async_execute() {
  // CHECK:      %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
  // CHECK:      %[[e0:.*]] = tfrt_gpu.event.create %[[ctx]]
  // CHECK:      %[[ch1:.*]] = tfrt_gpu.event.record %[[e0]], %arg1, %arg0
  // CHECK:      %[[t0:.*]]:2 = tfrt_test.do.async %[[ctx]], %[[e0]], %[[ch1]]
  // CHECK-SAME:    : (!tfrt_gpu.context, !tfrt_gpu.event, !tfrt.chain) ->
  // CHECK-SAME:      (!tfrt.chain, !tfrt_gpu.event)  {
  // %a0 has type !async.token, used as dependency in the second async.execute.
  %a0, %f0 = async.execute -> !async.value<!gpu.async.token> {
    // CHECK: %[[str0:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch2:.*]] = tfrt_gpu.stream.wait %[[str0]], %[[e0]], %[[ch1]]
    // CHECK: %[[e1:.*]] = tfrt_gpu.event.create %[[ctx]]
    // CHECK: %[[ch3:.*]] = tfrt_gpu.event.record %[[e1]], %[[str0]], %[[ch2]]
    %t0 = gpu.wait async
    // CHECK: tfrt.return %[[ch3]], %[[e1]] : !tfrt.chain, !tfrt_gpu.event
    async.yield %t0 : !gpu.async.token
  }
  // CHECK:      %[[t1:.*]]:2 = tfrt_test.do.async %[[t0]]#0, %[[t0]]#1, %[[ctx]]
  // CHECK-SAME:    : (!tfrt.chain, !tfrt_gpu.event, !tfrt_gpu.context) ->
  // CHECK-SAME:      (!tfrt.chain, !tfrt_gpu.event)  {
  // %a1 has type !async.token, unused.
  %a1, %f1 = async.execute [%a0] (
    %f0 as %t0 : !async.value<!gpu.async.token>
  ) -> !async.value<!gpu.async.token> {
    // CHECK: %[[str1:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch4:.*]] = tfrt_gpu.stream.wait %[[str1]], %[[t0]]#1, %[[t0]]#0
    // CHECK: %[[e2:.*]] = tfrt_gpu.event.create %[[ctx]]
    // CHECK: %[[ch5:.*]] = tfrt_gpu.event.record %[[e2]], %[[str1]], %[[ch4]]
    %t1 = gpu.wait async [%t0]
    // CHECK: tfrt.return %[[ch5]], %[[e2]] : !tfrt.chain, !tfrt_gpu.event
    async.yield %t1 : !gpu.async.token
  }
  // CHECK: %[[ch6:.*]] = tfrt.merge.chains %[[ch1]], %[[t1]]#0
  %t2 = gpu.wait async
  %t1 = async.await %f1 : !async.value<!gpu.async.token>
  // CHECK: %[[ch7:.*]] = tfrt_gpu.stream.wait %arg1, %[[t1]]#1, %[[ch6]]
  gpu.wait [%t1, %t2]
  // CHECK: tfrt.return %[[ch7]] : !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: @test_mem_cpy(
// CHECK-SAME:    %arg0: !tfrt.chain,
// CHECK-SAME:    %arg1: !tfrt_gpu.stream,
// CHECK-SAME:    %arg2: !tfrt_gpu.buffer
// CHECK-SAME:    %arg3: !tfrt_gpu.buffer
// CHECK-SAME:  ) -> !tfrt.chain {
func @test_mem_cpy(%arg0 : !tfrt_gpu.buffer, %arg1 : !tfrt_gpu.buffer) {
  %t0 = gpu.wait async
  %0 = builtin.unrealized_conversion_cast %arg0 : !tfrt_gpu.buffer to memref<32xi32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !tfrt_gpu.buffer to memref<32xi32>
  // CHECK-NEXT: %[[ch:.*]] = tfrt_gpu.mem.copy %arg2, %arg3, %arg1, %arg0 : !tfrt_gpu.buffer, !tfrt_gpu.buffer
  %t1 = gpu.memcpy async [%t0] %0, %1 : memref<32xi32>, memref<32xi32>
  gpu.wait [%t1]
  // CHECK-NEXT: tfrt.return %[[ch]] : !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: @test_mem_set(
// CHECK-SAME:    %arg0: !tfrt.chain,
// CHECK-SAME:    %arg1: !tfrt_gpu.stream,
// CHECK-SAME:    %arg2: !tfrt_gpu.buffer
// CHECK-SAME:  ) -> !tfrt.chain {
func @test_mem_set(%arg0 : !tfrt_gpu.buffer) {
  %t0 = gpu.wait async
  // CHECK-NEXT: %[[c0:.*]] = tfrt.constant.i32 0
  %c0 = tfrt.constant.i32 0
  %0 = builtin.unrealized_conversion_cast %arg0 : !tfrt_gpu.buffer to memref<32xi32>
  // CHECK-NEXT: %[[ch:.*]] = tfrt_gpu.mem.set %arg2, %[[c0]], %arg1, %arg0 : !tfrt_gpu.buffer, i32
  %t1 = gpu.memset async [%t0] %0, %c0 : memref<32xi32>, i32
  gpu.wait [%t1]
  // CHECK-NEXT: tfrt.return %[[ch]] : !tfrt.chain
  tfrt.return
}
