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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor 2>&1 | FileCheck %s --dump-input=fail

func @native_add(%a: i32, %b: i32) -> i32 attributes {hex.native}
func @native_async_add(%a: i32, %b: i32) -> i32 attributes {hex.native}

// CHECK-LABEL: --- Running 'call_native_add'
func @call_native_add() {
  %a = hex.constant.i32 1
  %b = hex.constant.i32 2

  %r = hex.call @native_add(%a, %b) : (i32, i32) -> i32

  %ch0 = hex.new.chain
  // CHECK: int32 = 3
  hex.print.i32 %r, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'call_native_async_add'
func @call_native_async_add() {
  %a = hex.constant.i32 1
  %b = hex.constant.i32 2

  %r = hex.call @native_async_add(%a, %b) : (i32, i32) -> i32

  %ch0 = hex.new.chain
  // CHECK: int32 = 3
  hex.print.i32 %r, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'call_native_add_with_unavailable_input'
func @call_native_add_with_unavailable_input() {
  %a = hex.constant.i32 1
  %b = hex.constant.i32 1
  %c = tfrt_test.do.async %a, %b : (i32, i32) -> (i32) {
    %y_res = hex.add.i32 %a, %b
    hex.return %y_res : i32
  }

  %r = hex.call @native_add(%a, %c) : (i32, i32) -> i32

  %ch0 = hex.new.chain
  // CHECK: int32 = 3
  hex.print.i32 %r, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'native_error'
// CHECK: something bad happened
func @native_error() -> i32 attributes {hex.native}
