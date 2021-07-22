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

// RUN: bef_executor_lite %s.bef 2>&1 | FileCheck %s --dump-input=fail

func private @native_add(%a: i32, %b: i32) -> i32 attributes {tfrt.native}
func private @native_async_add(%a: i32, %b: i32) -> i32 attributes {tfrt.native}

// CHECK-LABEL: --- Running 'call_native_add'
func @call_native_add() {
  %a = tfrt.constant.i32 1
  %b = tfrt.constant.i32 2

  %r = tfrt.call @native_add(%a, %b) : (i32, i32) -> i32

  %ch0 = tfrt.new.chain
  // CHECK: int32 = 3
  tfrt.print.i32 %r, %ch0

  tfrt.return
}

// CHECK-LABEL: --- Running 'call_native_async_add'
func @call_native_async_add() {
  %a = tfrt.constant.i32 1
  %b = tfrt.constant.i32 2

  %r = tfrt.call @native_async_add(%a, %b) : (i32, i32) -> i32

  %ch0 = tfrt.new.chain
  // CHECK: int32 = 3
  tfrt.print.i32 %r, %ch0

  tfrt.return
}

// CHECK-LABEL: --- Running 'call_native_add_with_unavailable_input'
func @call_native_add_with_unavailable_input() {
  %a = tfrt.constant.i32 1
  %b = tfrt.constant.i32 1
  %c = tfrt_test.do.async %a, %b : (i32, i32) -> (i32) {
    %y_res = tfrt.add.i32 %a, %b
    tfrt.return %y_res : i32
  }

  %r = tfrt.call @native_add(%a, %c) : (i32, i32) -> i32

  %ch0 = tfrt.new.chain
  // CHECK: int32 = 3
  tfrt.print.i32 %r, %ch0

  tfrt.return
}

// CHECK-LABEL: --- Running 'native_error'
// CHECK: something bad happened
func private @native_error() -> i32 attributes {tfrt.native}
