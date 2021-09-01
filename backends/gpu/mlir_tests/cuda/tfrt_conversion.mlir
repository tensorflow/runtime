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

// RUN: tfrt_gpu_opt %s \
// RUN:   -test-tfrt-conversion \
// RUN:   -allow-unregistered-dialect \
// RUN: | FileCheck %s

// CHECK-LABEL: test_unwrap_async
"test_unwrap_async"() ( {  // Not a function to avoid signature conversion.
  // CHECK: %[[p:.*]]:2 = "chain_and_stream"() : () -> (!tfrt.chain, !tfrt_gpu.stream)
  %p:2 = "chain_and_stream"() : () -> (!tfrt.chain, !tfrt_gpu.stream)
  %t0 = builtin.unrealized_conversion_cast %p#0, %p#1
          : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %t1 = "tfrt_gpu_conversion.async.execute"(%t0) ( {
  ^bb0(%ch0: !tfrt.chain, %stream: !tfrt_gpu.stream):
    // CHECK-NEXT: %[[ch1:.*]] = tfrt.merge.chains %[[p]]#0, %[[p]]#1 : !tfrt.chain, !tfrt_gpu.stream
    %ch1 = tfrt.merge.chains %ch0, %stream : !tfrt.chain, !tfrt_gpu.stream
    tfrt.return %ch1 : !tfrt.chain
  }) : (!gpu.async.token) -> (!gpu.async.token)
  // CHECK-NEXT: builtin.unrealized_conversion_cast %[[ch1]], %[[p]]#1 : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
}) : () -> ()

// CHECK-LABEL: @test_signature_rewrite
// CHECK-SAME: (%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream) -> !tfrt.chain
func @test_signature_rewrite() {
  // CHECK: tfrt.return %arg0 : !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: @test_erase_gpu_wait
func @test_erase_gpu_wait(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream) -> !tfrt.chain {
  // CHECK-NOT: gpu.wait
  // These two ops are folded away completely.
  %t0 = gpu.wait async
  gpu.wait [%t0]
  // CHECK: tfrt.return %arg0 : !tfrt.chain
  tfrt.return %arg0 : !tfrt.chain
}

// CHECK-LABEL: @test_async_execute
func @test_async_execute(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream) -> !tfrt.chain {
  // CHECK: %[[a0:.*]], %[[f0:.*]] = async.execute
  // CHECK-SAME: -> !async.value<!tfrt_gpu.event> {
  // %a0 has type !async.token, used as dependency in the second async.execute.
  %a0, %f0 = async.execute -> !async.value<!gpu.async.token> {
    // CHECK: %[[ch0:.*]] = tfrt.new.chain
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
    // CHECK: %[[e0:.*]] = tfrt_gpu.event.create %[[ctx]]
    // CHECK: %[[ch1:.*]] = tfrt_gpu.event.record %[[e0]], %arg1, %[[ch0]]
    // CHECK: %[[str0:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch2:.*]] = tfrt_gpu.stream.wait %[[str0]], %[[e0]], %[[ch1]]
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %[[str0]]
    // CHECK: %[[e1:.*]] = tfrt_gpu.event.create %[[ctx]]
    // CHECK: %[[ch3:.*]] = tfrt_gpu.event.record %[[e1]], %[[str0]], %[[ch2]]
    %t0 = gpu.wait async
    // CHECK: async.yield %[[e1]] : !tfrt_gpu.event
    async.yield %t0 : !gpu.async.token
  }
  // CHECK: %[[a1:.*]], %[[f1:.*]] = async.execute [%[[a0]]] (
  // %a1 has type !async.token, unused.
  %a1, %f1 = async.execute [%a0] (
    // CHECK-SAME: %[[f0]] as %[[e1:.*]]: !async.value<!tfrt_gpu.event>
    %f0 as %t0 : !async.value<!gpu.async.token>
  // CHECK-SAME: ) -> !async.value<!tfrt_gpu.event> {
  ) -> !async.value<!gpu.async.token> {
    // CHECK: %[[ch4:.*]] = tfrt.new.chain
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
    // CHECK: %[[str1:.*]] = tfrt_gpu.stream.create %[[ctx]]
    // CHECK: %[[ch5:.*]] = tfrt_gpu.stream.wait %[[str1]], %[[e1]], %[[ch4]]
    // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %[[str1]]
    // CHECK: %[[e2:.*]] = tfrt_gpu.event.create %[[ctx]]
    // CHECK: %[[ch6:.*]] = tfrt_gpu.event.record %[[e2]], %[[str1]], %[[ch5]]
    %t1 = gpu.wait async [%t0]
    // CHECK: async.yield %[[e2]] : !tfrt_gpu.event
    async.yield %t1 : !gpu.async.token
  }
  // CHECK: %[[e2:.*]] = async.await %[[f1]] : !async.value<!tfrt_gpu.event>
  %t1 = async.await %f1 : !async.value<!gpu.async.token>
  // CHECK: %[[chx:.*]] = tfrt_gpu.stream.wait %arg1, %[[e2]], %arg0
  gpu.wait [%t1]
  // CHECK: tfrt.return %[[chx]] : !tfrt.chain
  tfrt.return %arg0 : !tfrt.chain
}

