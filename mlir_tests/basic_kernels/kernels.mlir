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

// RUN: tfrt_opt -allow-unregistered-dialect %s | tfrt_opt -allow-unregistered-dialect | tfrt_opt -allow-unregistered-dialect | FileCheck %s

func @if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  // CHECK: [[RES:%[0-9]+]] = tfrt.if %arg0, %arg1, %arg2 : (i32, i32) -> (i32) {
  %res = tfrt.if %cond, %v1, %v2 : (i32, i32) -> i32 {
    // CHECK-NEXT: tfrt.return %arg1
    tfrt.return %v1 : i32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: tfrt.return %arg2
    tfrt.return %v2 : i32
  }

  // CHECK: tfrt.if %arg0 : () -> () {
  tfrt.if %cond : () -> () {
    tfrt.return
  }

  // CHECK: tfrt.return [[RES]]
  tfrt.return %res : i32
}

// CHECK-LABEL: func @repeat(%arg0: i32, %arg1: f32) -> i32 {
func @repeat(%arg0: i32, %arg1: f32) -> i32 {
  // CHECK: [[RES:%[0-9]+]]:2 = tfrt.repeat.i32 %arg0, %arg0, %arg1 : i32, f32 {
  %res1, %res2 = tfrt.repeat.i32 %arg0, %arg0, %arg1 : i32, f32 {

    // CHECK-NEXT: "use"(%arg0, %arg1)
    "use"(%arg0, %arg1) : (i32, f32) -> ()

    tfrt.return %arg0, %arg1 : i32, f32
  }

  // Zero argument loop.

  // CHECK: tfrt.repeat.i32 %arg0 {
  tfrt.repeat.i32 %arg0 {
    tfrt.return
  }

  // CHECK: tfrt.return [[RES]]#0
  tfrt.return %res1 : i32
}

// CHECK-LABEL: func @return(
func @return(%arg: i32) -> i32 {
  // CHECK: tfrt.return %{{.*}} : i32
  tfrt.return %arg : i32
}

// CHECK-LABEL: func @constant(
func @constant() {
  // CHECK-NEXT: %0 = tfrt.constant.i1 false
  %a = tfrt.constant.i1 false
  // CHECK-NEXT: %1 = tfrt.constant.i32 41
  %b = tfrt.constant.i32 41
  // CHECK-NEXT: %2 = tfrt.constant.f32 1.000000e-01
  %c = tfrt.constant.f32 0.1
  // CHECK-NEXT: %3 = tfrt.constant.ui32 4095
  %d = tfrt.constant.ui32 4095
  tfrt.return
}
