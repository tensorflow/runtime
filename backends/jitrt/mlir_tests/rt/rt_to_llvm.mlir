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

// RUN: jitrt_opt %s --split-input-file --rt-to-llvm | FileCheck %s

// CHECK: func @pass_context(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @pass_context(%arg0: !rt.kernel_context) {
  func.return
}

// -----

// CHECK: func @set_output(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @set_output(%arg0: !rt.kernel_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  // CHECK: %[[LLVM_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEMREF]]
  %0 = memref.alloc() : memref<f32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK: %[[RES_PTR:.*]] = call @runtimeGetResultStorage(%[[CTX]], %[[C0]])
  // CHECK: %[[LLVM_PTR:.*]] = llvm.bitcast %[[RES_PTR]]
  // CHECK: llvm.store %[[LLVM_MEMREF]], %[[LLVM_PTR]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  func.return
}

// -----

// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR0:.*]]("Failed precondition #0\00")
// CHECK-DAG: llvm.mlir.global {{.*}} @[[ERR1:.*]]("Failed precondition #1\00")

// CHECK: func @set_error(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>
// CHECK: )
func.func @set_error(%arg0: !rt.kernel_context) {
  // CHECK: %[[ADDR0:.*]] = llvm.mlir.addressof @[[ERR0]]
  // CHECK: %[[PTR0:.*]] = llvm.bitcast %[[ADDR0]] {{.*}} to !llvm.ptr<i8>
  // CHECK: call @runtimeSetError(%[[CTX]], %[[PTR0]])
  rt.set_error %arg0, "Failed precondition #0"
  // CHECK: %[[ADDR1:.*]] = llvm.mlir.addressof @[[ERR1]]
  // CHECK: %[[PTR1:.*]] = llvm.bitcast %[[ADDR1]] {{.*}} to !llvm.ptr<i8>
  // CHECK: call @runtimeSetError(%[[CTX]], %[[PTR1]])
  rt.set_error %arg0, "Failed precondition #1"
  func.return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @[[REDUCE:.*]]("f32_reduce\00")
// CHECK-DAG: llvm.mlir.global internal constant @[[INIT:.*]]("init\00")

// CHECK: func @custom_call(
// CHECK:   %[[CTX:.*]]: !llvm.ptr<i8>,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: )
func.func @custom_call(%arg0: !rt.kernel_context, %arg1: memref<?xf32>) {
  // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG]]
  // CHECK-SAME: : memref<?xf32> to !llvm.struct

  // CHECK: %[[CALLEE_ADDR:.*]] = llvm.mlir.addressof @[[REDUCE]]
  // CHECK: %[[CALLEE:.*]] = llvm.bitcast %[[CALLEE_ADDR]]

  // Arguments encoding:
  // CHECK: llvm.mlir.undef : !llvm.struct<(i64, i64, ptr<i8>)>
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[ARGS:.*]] = llvm.alloca %[[C3]] x !llvm.ptr<i8>

  // Attributes encoding:

  // CHECK: llvm.mlir.addressof @[[INIT]] : !llvm.ptr<array<5 x i8>>
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[ATTRS:.*]] = llvm.alloca %[[C4]] x !llvm.ptr<i8>

  // CHECK: %[[STATUS:.*]] = call @runtimeCustomCall(%[[CALLEE]],
  // CHECK-SAME:                                     %[[ARGS]],
  // CHECK-SAME:                                     %[[ATTRS]])
  // CHECK: cf.assert %[[STATUS]], "oops"
  %status = rt.custom_call "f32_reduce"(%arg1)
              { init = 1.0 : f32  }
              : (memref<?xf32>) -> ()

  %ok = rt.is_ok %status
  cf.assert %ok, "oops"

  func.return
}
