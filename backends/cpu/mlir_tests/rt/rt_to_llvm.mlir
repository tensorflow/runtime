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

// RUN: cpurt_opt %s --rt-to-llvm | FileCheck %s

// CHECK: func @pass_context(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func @pass_context(%arg0: !rt.kernel_context) {
  return
}

// CHECK: func @set_output(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func @set_output(%arg0: !rt.kernel_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  // CHECK: %[[LLVM_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEMREF]]
  %0 = memref.alloc() : memref<f32>
  // CHECK: %[[C0:.*]] = constant 0 : i64
  // CHECK: %[[RES_PTR:.*]] = call @runtimeGetResultStorage(%[[CTX]], %[[C0]])
  // CHECK: %[[LLVM_PTR:.*]] = llvm.bitcast %[[RES_PTR]]
  // CHECK: llvm.store %[[LLVM_MEMREF]], %[[LLVM_PTR]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  return
}

// CHECK: func @set_error(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func @set_error(%arg0: !rt.kernel_context) {
  // CHECK: call @runtimeSetError(%[[CTX]])
  rt.set_error %arg0, "Failed precondition"
  return
}
