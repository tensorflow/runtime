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

// RUN: tfrt_gpu_opt %s -async-to-tfrt | FileCheck %s

// CHECK-LABEL: @async_execute
func @async_execute(%arg0 : i32) -> i32 {
  // CHECK: %[[i0:.*]] = tfrt_test.do.async %arg0 : (i32) -> (i32)
  // %a0 has type !async.token, used as dependency in the second async.execute.
  %a0, %f0 = async.execute -> !async.value<i32> {
    // CHECK: tfrt.return %arg0 : i32
    async.yield %arg0 : i32
  }
  // CHECK: %[[i1:.*]] = tfrt_test.do.async %[[i0]] : (i32) -> (i32)
  // %a1 has type !async.token, unused.
  %a1, %f1 = async.execute [%a0] (
    %f0 as %i0 : !async.value<i32>
  ) -> !async.value<i32> {
    // CHECK: tfrt.return %[[i0]] : i32
    async.yield %i0 : i32
  }
  %i1 = async.await %f1 : !async.value<i32>
  // CHECK: tfrt.return %[[i1]] : i32
  tfrt.return %i1 : i32
}
