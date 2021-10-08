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

// RUN: cpurt_opt %s | FileCheck %s

// CHECK-LABEL: func @pass_context(
// CHECK:  %[[CTX:.*]]: !rt.kernel_context
func @pass_context(%arg0: !rt.kernel_context) {
  return
}

// CHECK-LABEL: func @set_output(
// CHECK:  %[[CTX:.*]]: !rt.kernel_context
func @set_output(%arg0: !rt.kernel_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %0 = memref.alloc() : memref<f32>
  // CHECK: rt.set_output %[[CTX]], 0, %[[MEMREF]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  return
}

// CHECK-LABEL: func @set_error(
// CHECK:  %[[CTX:.*]]: !rt.kernel_context
func @set_error(%arg0: !rt.kernel_context) {
  // CHECK: rt.set_error %[[CTX]], "Failed precondition"
  rt.set_error %arg0, "Failed precondition"
  return
}
