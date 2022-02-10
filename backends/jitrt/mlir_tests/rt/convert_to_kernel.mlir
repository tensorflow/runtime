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

// RUN: jitrt_opt %s --rt-to-kernel-function | FileCheck %s

// CHECK: func @single_result(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) {
func @single_result(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { jitrt.entrypoint } {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0 : memref<?xf32>
}

// CHECK: func @two_results(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) {
func @two_results(%arg0: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>)
  attributes { jitrt.entrypoint } {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: rt.set_output %[[CTX]], 1, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0, %arg0 : memref<?xf32>, memref<?xf32>
}

// CHECK: func @not_an_entrypoint(
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) -> memref<?xf32> {
func @not_an_entrypoint(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NOT: rt.set_output
  // CHECK: return %[[ARG]]
  return %arg0 : memref<?xf32>
}

// CHECK: func @assert_to_error(
// CHECK:   %[[CTX:.*]]: !rt.kernel_context,
// CHECK:   %[[ASSERT:.*]]: i1
// CHECK: ) {
func @assert_to_error(%arg0: i1) attributes { jitrt.entrypoint } {
  // CHECK: cond_br %[[ASSERT]], ^[[OK:.*]], ^[[ERR:.*]]
  // CHECK: ^[[OK]]:
  // CHECK:   return
  // CHECK: ^[[ERR]]:
  // CHECK:   rt.set_error %[[CTX]], "Failed precondition"
  // CHECK:   return
  cf.assert %arg0, "Failed precondition"
  return
}
