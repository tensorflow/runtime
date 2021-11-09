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

// RUN: tfrt_gpu_opt %s -async-tfrt-streamify | FileCheck %s

// CHECK-LABEL: @to_chain_and_event
func @to_chain_and_event(
  %arg0 : !async.token,
  %arg1 : !async.value<!gpu.async.token>
) {

  // CHECK: %[[args:.*]]:2 = builtin.unrealized_conversion_cast %arg1
  // CHECK-SAME: : !async.value<!gpu.async.token>
  // CHECK-SAME:   to !async.value<!tfrt.chain>, !async.value<!tfrt_gpu.event>

  // CHECK: %token, %results:2 = async.execute [%arg0] (
  // CHECK-SAME:   %[[args]]#0 as %arg2: !async.value<!tfrt.chain>,
  // CHECK-SAME:   %[[args]]#1 as %arg3: !async.value<!tfrt_gpu.event>
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   !async.value<!tfrt.chain>, !async.value<!tfrt_gpu.event>
  // CHECK-SAME: ) {
  %token, %results = async.execute [%arg0] (
   %arg1 as %t0: !async.value<!gpu.async.token>
  ) -> !async.value<!gpu.async.token> {
    // CHECK: %[[cast1:.*]] = builtin.unrealized_conversion_cast %arg2, %arg3
    // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.event to !gpu.async.token

    // CHECK: %[[cast2:.*]]:2 = builtin.unrealized_conversion_cast %[[cast1]]
    // CHECK-SAME: : !gpu.async.token to !tfrt.chain, !tfrt_gpu.event

    // CHECK: async.yield %[[cast2]]#0, %[[cast2]]#1
    // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.event
    async.yield %t0 : !gpu.async.token
  }

  // CHECK: %[[chain:.*]] = async.await %results#0 : !async.value<!tfrt.chain>
  // CHECK: %[[event:.*]] = async.await %results#1 : !async.value<!tfrt_gpu.event>
  // CHECK: builtin.unrealized_conversion_cast %[[chain]], %[[event]]
  // CHECK-SAME: !tfrt.chain, !tfrt_gpu.event to !gpu.async.token
  %t1 = async.await %results : !async.value<!gpu.async.token>

  tfrt.return
}
