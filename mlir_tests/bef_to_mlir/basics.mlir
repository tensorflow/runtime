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

// RUN: tfrt_translate --bef-to-mlir %s.bef | tfrt_opt | FileCheck %s

// CHECK-LABEL: func @basic.argument
func @basic.argument(%a : i32) -> i32 {
  // CHECK: tfrt.return {{%.*}} : i32
  tfrt.return %a : i32
}

// CHECK-LABEL: func @basic.add() -> i32
func @basic.add() -> i32 {
  // CHECK-NEXT: [[REG0:%.*]] = tfrt.constant.i32 42
  // CHECK-NEXT: [[REG1:%.*]] = tfrt.constant.i32 17
  // CHECK-NEXT: [[REG2:%.*]] = tfrt.add.i32 [[REG0]], [[REG1]]
  // CHECK-NEXT: tfrt.return [[REG2]] : i32

  %x = tfrt.constant.i32 42
  %y = tfrt.constant.i32 17
  %z = tfrt.add.i32 %x, %y
  tfrt.return %z : i32
}

// CHECK-LABEL: func @basic.addarg
func @basic.addarg(%a : i32, %b : i32) -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = tfrt.add.i32 {{%.*}}, {{%.*}}
  // CHECK-NEXT: tfrt.return [[REG]] : i32
  %c = tfrt.add.i32 %a, %b
  tfrt.return %c : i32
}

// CHECK-LABEL: func @basic.call() -> i32
func @basic.call() -> i32 {
  // CHECK-NEXT: [[REG0:%.*]] = tfrt.constant.i32 42
  // CHECK-NEXT: [[REG1:%.*]] = tfrt.constant.i32 17
  // CHECK-NEXT: [[REG2:%.*]] = tfrt.call @basic.addarg([[REG0]], [[REG1]])  : (i32, i32) -> i32
  // CHECK-NEXT: tfrt.return [[REG2]] : i32

  %x = tfrt.constant.i32 42
  %y = tfrt.constant.i32 17
  %z = tfrt.call @basic.addarg(%x, %y) : (i32, i32) -> i32
  tfrt.return %z : i32
}

// CHECK-LABEL: func @controlflow_if({{%.*}}: i1, {{%.*}}: i32, {{%.*}}: i32) -> i32 {
func @controlflow_if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  // CHECK-NEXT: [[RES:%.*]] = tfrt.if {{%.*}}, [[REG1:%.*]], [[REG2:%.*]] : (i32, i32) -> (i32) {
  %res = tfrt.if %cond, %v1, %v2 : (i32, i32) -> (i32) {
  // CHECK-NEXT:    tfrt.return [[REG1]] : i32
    tfrt.return %v1 : i32
  // CHECK-NEXT:  } else {
  } else {
  // CHECK-NEXT:    tfrt.return [[REG2]] : i32
    tfrt.return %v2 : i32
  }

  // CHECK: tfrt.return [[RES]] : i32
  tfrt.return %res : i32
}

// CHECK-LABEL: func @basic.print
func @basic.print(%x: i32, %y: i32, %z: i32) {
  // CHECK-NEXT: [[REG:%.*]] = tfrt.new.chain
  %ch0 = tfrt.new.chain

  // CHECK-NEXT: tfrt.print.i32 {{%.*}} [[REG]]
  // CHECK-NEXT: tfrt.print.i32 {{%.*}} [[REG]]
  // CHECK-NEXT: tfrt.print.i32 {{%.*}} [[REG]]
  tfrt.print.i32 %x, %ch0
  tfrt.print.i32 %y, %ch0
  tfrt.print.i32 %z, %ch0

  // CHECK-NEXT: tfrt.return
  tfrt.return
}

// CHECK-LABEL: func @call.non_strict()
func @call.non_strict() {
  %c1 = tfrt.constant.i32 1
  %v2 = "tfrt.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "tfrt.async_add.i32"(%c1, %v2) : (i32, i32) -> i32

  // CHECK: tfrt.call @basic.print({{%.*}}, {{%.*}}, {{%.*}})  : (i32, i32, i32) -> ()
  tfrt.call @basic.print(%c1, %v2, %v3) : (i32, i32, i32) -> ()

  %c5 = tfrt.constant.i32 5
  %ch0 = tfrt.new.chain
  tfrt.print.i32 %c5, %ch0
  tfrt.return
}
