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

// RUN: tfrt_translate -mlir-to-bef %s | tfrt_translate --bef-to-mlir | tfrt_opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @basic.argument
func @basic.argument(%a : i32) -> i32 {
  // CHECK: hex.return {{%.*}} : i32
  hex.return %a : i32
}

// CHECK-LABEL: func @basic.add() -> i32
func @basic.add() -> i32 {
  // CHECK-NEXT: [[REG0:%.*]] = hex.constant.i32 42
  // CHECK-NEXT: [[REG1:%.*]] = hex.constant.i32 17
  // CHECK-NEXT: [[REG2:%.*]] = hex.add.i32 [[REG0]], [[REG1]]
  // CHECK-NEXT: hex.return [[REG2]] : i32

  %x = hex.constant.i32 42
  %y = hex.constant.i32 17
  %z = hex.add.i32 %x, %y
  hex.return %z : i32
}

// CHECK-LABEL: func @basic.addarg
func @basic.addarg(%a : i32, %b : i32) -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = hex.add.i32 {{%.*}}, {{%.*}}
  // CHECK-NEXT: hex.return [[REG]] : i32
  %c = hex.add.i32 %a, %b
  hex.return %c : i32
}

// CHECK-LABEL: func @basic.call() -> i32
func @basic.call() -> i32 {
  // CHECK-NEXT: [[REG0:%.*]] = hex.constant.i32 42
  // CHECK-NEXT: [[REG1:%.*]] = hex.constant.i32 17
  // CHECK-NEXT: [[REG2:%.*]] = hex.call @basic.addarg([[REG0]], [[REG1]])  : (i32, i32) -> i32
  // CHECK-NEXT: hex.return [[REG2]] : i32

  %x = hex.constant.i32 42
  %y = hex.constant.i32 17
  %z = hex.call @basic.addarg(%x, %y) : (i32, i32) -> i32
  hex.return %z : i32
}

// CHECK-LABEL: func @controlflow_if({{%.*}}: i1, {{%.*}}: i32, {{%.*}}: i32) -> i32 {
func @controlflow_if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  // CHECK-NEXT: [[RES:%.*]] = hex.if {{%.*}}, [[REG1:%.*]], [[REG2:%.*]] : (i32, i32) -> (i32) {
  %res = hex.if %cond, %v1, %v2 : (i32, i32) -> (i32) {
  // CHECK-NEXT:    hex.return [[REG1]] : i32
    hex.return %v1 : i32
  // CHECK-NEXT:  } else {
  } else {
  // CHECK-NEXT:    hex.return [[REG2]] : i32
    hex.return %v2 : i32
  }

  // CHECK: hex.return [[RES]] : i32
  hex.return %res : i32
}

// CHECK-LABEL: func @basic.print
func @basic.print(%x: i32, %y: i32, %z: i32) {
  // CHECK-NEXT: [[REG:%.*]] = hex.new.chain
  %ch0 = hex.new.chain

  // CHECK-NEXT: hex.print.i32 {{%.*}} [[REG]]
  // CHECK-NEXT: hex.print.i32 {{%.*}} [[REG]]
  // CHECK-NEXT: hex.print.i32 {{%.*}} [[REG]]
  hex.print.i32 %x, %ch0
  hex.print.i32 %y, %ch0
  hex.print.i32 %z, %ch0

  // CHECK-NEXT: hex.return
  hex.return
}

// CHECK-LABEL: func @call.non_strict()
func @call.non_strict() {
  %c1 = hex.constant.i32 1
  %v2 = "hex.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "hex.async_add.i32"(%c1, %v2) : (i32, i32) -> i32

  // CHECK: hex.call @basic.print({{%.*}}, {{%.*}}, {{%.*}})  : (i32, i32, i32) -> ()
  hex.call @basic.print(%c1, %v2, %v3) : (i32, i32, i32) -> ()

  %c5 = hex.constant.i32 5
  %ch0 = hex.new.chain
  hex.print.i32 %c5, %ch0
  hex.return
}
