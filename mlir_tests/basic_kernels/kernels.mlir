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

// RUN: tfrt_opt -allow-unregistered-dialect %s | tfrt_opt -allow-unregistered-dialect | tfrt_opt -allow-unregistered-dialect | FileCheck %s --dump-input=fail

func @if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  // CHECK: [[RES:%[0-9]+]] = hex.if %arg0, %arg1, %arg2 : (i32, i32) -> (i32) {
  %res = hex.if %cond, %v1, %v2 : (i32, i32) -> i32 {
    // CHECK-NEXT: hex.return %arg1
    hex.return %v1 : i32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: hex.return %arg2
    hex.return %v2 : i32
  }

  // CHECK: hex.if %arg0 : () -> () {
  hex.if %cond : () -> () {
    hex.return
  }

  // CHECK: hex.return [[RES]]
  hex.return %res : i32
}

// CHECK-LABEL: func @repeat(%arg0: i32, %arg1: f32) -> i32 {
func @repeat(%arg0: i32, %arg1: f32) -> i32 {
  // CHECK: [[RES:%[0-9]+]]:2 = hex.repeat.i32 %arg0, %arg0, %arg1 : i32, f32 {
  %res1, %res2 = hex.repeat.i32 %arg0, %arg0, %arg1 : i32, f32 {

    // CHECK-NEXT: "use"(%arg0, %arg1)
    "use"(%arg0, %arg1) : (i32, f32) -> ()

    hex.return %arg0, %arg1 : i32, f32
  }

  // Zero argument loop.

  // CHECK: hex.repeat.i32 %arg0 {
  hex.repeat.i32 %arg0 {
    hex.return
  }

  // CHECK: hex.return [[RES]]#0
  hex.return %res1 : i32
}

// CHECK-LABEL: func @return(
func @return(%arg: i32) -> i32 {
  // CHECK: hex.return %{{.*}} : i32
  hex.return %arg : i32
}

// CHECK-LABEL: func @constant(
func @constant() {
  // CHECK-NEXT: %0 = hex.constant.i1 0
  %a = hex.constant.i1 0
  // CHECK-NEXT: %1 = hex.constant.i32 41
  %b = hex.constant.i32 41
  // CHECK-NEXT: %2 = hex.constant.f32 1.000000e-01
  %c = hex.constant.f32 0.1
  hex.return
}
