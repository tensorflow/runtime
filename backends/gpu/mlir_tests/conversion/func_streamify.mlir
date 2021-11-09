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

// RUN: tfrt_gpu_opt %s -func-tfrt-streamify | FileCheck %s

// CHECK-LABEL: @add_chain_and_stream(
// CHECK-SAME:    %arg0: !tfrt.chain,
// CHECK-SAME:    %arg1: !tfrt_gpu.stream,
// CHECK-SAME:    %arg2: f32 {tfrt_gpu.attr = 42 : i32}
// CHECK-SAME:  ) -> !tfrt.chain {
func @add_chain_and_stream(%arg0 : f32 {tfrt_gpu.attr = 42 : i32}) {
  // CHECK: %[[t0:.*]] = builtin.unrealized_conversion_cast %arg0, %arg1
  // CHECK-SAME: : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  // CHECK: %[[t1:.*]] = gpu.wait async [%[[t0]]]
  // CHECK: %[[t2:.*]] = gpu.wait async [%[[t1]]]
  %t0 = gpu.wait async
  // CHECK: %[[t3:.*]] = gpu.wait async [%[[t2]]]
  // CHECK: %[[cast:.*]]:2 = builtin.unrealized_conversion_cast %[[t3]]
  // CHECK-SAME: : !gpu.async.token to !tfrt.chain, !tfrt_gpu.stream
  gpu.wait [%t0]
  // CHECK: tfrt.return %[[cast]]#0 : !tfrt.chain
  tfrt.return
}
