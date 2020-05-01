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

// RUN: tfrt_translate -mlir-to-bef %s | tfrt_translate -bef-to-mlir | tfrt_opt -allow-unregistered-dialect | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @bool.constant() -> i1
func @bool.constant() -> i1 {
  // CHECK-NEXT: [[REG:%.*]] = hex.constant.bool true
  // CHECK-NEXT: hex.return [[REG]] : i1

  %x = hex.constant.bool true
  hex.return %x : i1
}

// CHECK-LABEL: func @integer1.constant() -> i1
func @integer1.constant() -> i1 {
  // CHECK-NEXT: [[REG:%.*]] = hex.constant.i1 1
  // CHECK-NEXT: hex.return [[REG]] : i1

  %x = hex.constant.i1 1
  hex.return %x : i1
}

// CHECK-LABEL: func @integer32.constant() -> i32
func @integer32.constant() -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = hex.constant.i32 42
  // CHECK-NEXT: hex.return [[REG]] : i32

  %x = hex.constant.i32 42
  hex.return %x : i32
}

// CHECK-LABEL: func @integer64.constant() -> i64
func @integer64.constant() -> i64 {
  // CHECK-NEXT: [[REG:%.*]] = hex.constant.i64 42
  // CHECK-NEXT: hex.return [[REG]] : i64

  %x = hex.constant.i64 42
  hex.return %x : i64
}

// CHECK-LABEL: func @float32.constant() -> f32
func @float32.constant() -> f32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"() {value = {{.*}} : f32} : () -> f32
  // CHECK-NEXT: hex.return [[REG]] : f32

  %x = "simple.op"() {value = -0.1 : f32} : () -> f32
  hex.return %x : f32
}

// CHECK-LABEL: func @string.constant() -> f32
func @string.constant() -> f32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"()
  // CHECK-SAME: {value = "string constant"} : () -> f32
  // CHECK-NEXT: hex.return [[REG]] : f32

  %x = "simple.op"() {value = "string constant"} : () -> f32
  hex.return %x : f32
}

// CHECK-LABEL: func @array.constant() -> i32
func @array.constant() -> i32 {
  // CHECK: {value = [1 : i32, 2 : i32]} : () -> i32
  // CHECK: {value = [i32, f32]} : () -> i32
  // CHECK: {value = [true, false]} : () -> i32
  // CHECK: {value = [dense<0> : tensor<4xi64>, dense<1> : tensor<4xi64>]} : () -> i32

  %a = "simple.op"() {value = [1 : i32, 2 : i32]} : () -> i32
  %b = "simple.op"() {value = [i32, f32]} : () -> i32
  %c = "simple.op"() {value = [true, false]} : () -> i32
  %d = "simple.op"() {value = [dense<0> : tensor<4xi64>, dense<1> : tensor<4xi64>]} : () -> i32
  hex.return %d : i32
}

// CHECK-LABEL: func @aggregate.constant() -> i32
func @aggregate.constant() -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"()
  // CHECK-SAME: {value = [1 : i32, [2 : i32, "string"]]} : () -> i32
  // CHECK-NEXT: hex.return [[REG]] : i32

  %x = "simple.op"() {value = [1: i32, [2: i32, "string"]]} : () -> i32
  hex.return %x : i32
}

// CHECK-LABEL: func @type.attribute() -> (i32, i32)
func @type.attribute() -> (i32, i32) {
  // CHECK-NEXT: [[REG0:%.*]] = "get_width"() {value = i32} : () -> i32
  // CHECK-NEXT: [[REG1:%.*]] = "get_width"() {value = f64} : () -> i32
  // CHECK-NEXT: hex.return [[REG0]], [[REG1]] : i32, i32

  %x = "get_width"() {value = i32} : () -> i32
  %y = "get_width"() {value = f64} : () -> i32
  hex.return %x, %y : i32, i32
}

// CHECK-LABEL: func @dense_elements.constant() -> i32
func @dense_elements.constant() -> i32 {

  // CHECK-NEXT: [[REG0:%.*]] = "simple.op"()
  // CHECK-SAME:   {value = dense<[]> : tensor<0xf32>} : () -> i32
  %x0 = "simple.op"() {value = dense<[]> : tensor<0xf32>} : () -> i32

  // CHECK-NEXT: [[REG1:%.*]] = "simple.op"()
  // CHECK-SAME: {value = dense<1> : tensor<2x3xi64>} : () -> i32
  %x1 = "simple.op"() {value = dense<1> : tensor<2x3xi64>} : () -> i32

  // CHECK-NEXT: [[REG2:%.*]] = "simple.op"()
  // CHECK-SAME: {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> i32
  %x2 = "simple.op"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> i32

  // CHECK-NEXT: hex.return [[REG2]] : i32
  hex.return %x2 : i32
}

// CHECK-LABEL: @shape_attr
func @shape_attr() {
  // CHECK: #corert.shape<2x?x3>
  "simple.op"() {shape = #corert.shape<2x?x3>} : () -> ()
  hex.return
}
