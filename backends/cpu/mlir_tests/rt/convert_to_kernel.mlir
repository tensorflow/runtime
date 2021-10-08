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

// RUN: cpurt_opt %s --rt-to-kernel-function | FileCheck %s

// CHECK: func @single_result(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[RET:.*]]: memref<?xf32>
// CHECK: ) {
func @single_result(%arg0: !rt.kernel_context, %arg1: memref<?xf32>)
    -> memref<?xf32> {
  // CHECK: rt.set_output %[[CTX]], 0, %[[RET]] : memref<?xf32>
  // CHECK: return
  return %arg1 : memref<?xf32>
}

// CHECK: func @two_results(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[RET:.*]]: memref<?xf32>
// CHECK: ) {
func @two_results(%arg0: !rt.kernel_context, %arg1: memref<?xf32>)
    -> (memref<?xf32>, memref<?xf32>) {
  // CHECK: rt.set_output %[[CTX]], 0, %[[RET]] : memref<?xf32>
  // CHECK: rt.set_output %[[CTX]], 1, %[[RET]] : memref<?xf32>
  // CHECK: return
  return %arg1, %arg1 : memref<?xf32>, memref<?xf32>
}

// CHECK: func @invalid_position(
// CHECK:   %[[RET:.*]]: memref<?xf32>,
// CHECK:   %[[CTX:.*]]: !rt.kernel_context
// CHECK: ) -> memref<?xf32> {
func @invalid_position(%arg0: memref<?xf32>, %arg1: !rt.kernel_context)
    -> memref<?xf32> {
  // CHECK-NOT: rt.set_output
  // CHECK: return %[[RET]]
  return %arg0 : memref<?xf32>
}

// CHECK: func @assert_to_error(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[ASSERT:.*]]: i1
// CHECK: ) {
func @assert_to_error(%arg0: !rt.kernel_context, %arg1: i1) {
  // CHECK: cond_br %[[ASSERT]], ^[[OK:.*]], ^[[ERR:.*]]
  // CHECK: ^[[OK]]:
  // CHECK:   return
  // CHECK: ^[[ERR]]:
  // CHECK:   rt.set_error %[[CTX]], "Failed precondition"
  // CHECK:   return
  assert %arg1, "Failed precondition"
  return
}
