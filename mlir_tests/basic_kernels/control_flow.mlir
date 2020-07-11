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

// RUN: tfrt_translate --mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail
// RUN: tfrt_translate --mlir-to-bef %s | bef_executor -work_queue_type=mstd | FileCheck %s --dump-input=fail

// This function is just a select: "cond ? v1 : v2", exercising hex.if with
// result values.
func @controlflow_if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  %res = hex.if %cond, %v1, %v2 : (i32, i32) -> (i32) {
    hex.return %v1 : i32
  } else {
    hex.return %v2 : i32
  }

  hex.return %res : i32
}

// CHECK-LABEL: --- Running 'if_test_0'
func @if_test_0() {
  %ch0 = hex.new.chain

  %a = hex.constant.i32 45
  %b = hex.constant.i32 129

  %true = hex.constant.i1 1

  // CHECK-NEXT: int32 = 45
  %x = hex.call @controlflow_if(%true, %a, %b) : (i1, i32, i32) -> (i32)
  %ch1 = hex.print.i32 %x, %ch0

  %false = hex.constant.i1 0

  // CHECK-NEXT: int32 = 129
  %y = hex.call @controlflow_if(%false, %a, %b) : (i1, i32, i32) -> (i32)
  %ch2 = hex.print.i32 %y, %ch1

  hex.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat2'
func @controlflow_repeat2() {
  %count = hex.constant.i32 4
  %ch0 = hex.new.chain

  %v0 = hex.constant.i32 1

  // This loop properly passes around the control chain to guarantee proper
  // sequencing of the print calls.
  //
  // The body of a repeat is a function call.  The "call" to it is passed the
  // %ch0 and %v0 values, the body of it receives them as %c1/%v1.
  %ch3:2 = hex.repeat.i32 %count, %ch0, %v0 : !hex.chain, i32 {
    %ch2 = hex.print.i32 %v0, %ch0
    %v2 = hex.add.i32 %v0, %v0
    hex.return %ch2, %v2 : !hex.chain, i32
  }

  // CHECK-NEXT: int32 = 1
  // CHECK-NEXT: int32 = 2
  // CHECK-NEXT: int32 = 4
  // CHECK-NEXT: int32 = 8

  %x = hex.constant.i32 -1

  // CHECK-NEXT: hello host executor!
  %ch4 = "tfrt_test.print_hello"(%ch3#0) : (!hex.chain) -> !hex.chain

  // CHECK-NEXT: int32 = 16
  hex.print.i32 %ch3#1, %ch4

  hex.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat5'
func @controlflow_repeat5() {
  %count = hex.constant.i32 5
  hex.repeat.i32 %count {
    %ch0 = hex.new.chain
    "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain

    %x = hex.constant.i32 0
    hex.print.i32 %x, %ch0

    hex.return
  }

  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0

  %x = hex.constant.i32 -1
  %ch1 = hex.new.chain

  // CHECK-NEXT: int32 = -1
  hex.print.i32 %x, %ch1

  hex.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat_large'
func @controlflow_repeat_large() {
  %count = hex.constant.i32 1000
  %v0 = hex.constant.i32 42

  //TODO(xldrx): Fix the race condition when there are multiple worker threads.
  %sum = hex.repeat.i32 %count, %v0: i32 {
    %one = hex.constant.i32 1
    %v2 = "hex.async_add.i32"(%v0, %one) : (i32, i32) -> i32
    hex.return %v2: i32
  }

  %ch0 = hex.new.chain
  // CHECK-NEXT: int32 = 1042
  hex.print.i32 %sum, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat_cancel'
func @controlflow_repeat_cancel() -> i32 {
  %count = hex.constant.i32 5
  %index = hex.constant.i32 0

  %repeat_result = hex.repeat.i32 %count, %index : i32 {
    %ch0 = hex.new.chain
    hex.print.i32 %index, %ch0

    %one = hex.constant.i32 1
    %cond = "hex.lessequal.i32"(%one, %index) : (i32, i32) -> (i1)

    // Cancel when the loop index reaches 1.
    %x = hex.if %cond, %one : (i32) -> i32 {
      %ch1 = hex.new.chain
      %x, %ch2 = "tfrt_test.cancel"(%ch1) : (!hex.chain) -> (i32, !hex.chain)
      hex.return %x : i32
    } else {
      hex.return %one : i32
    }

    %next_index = hex.add.i32 %index, %x
    hex.return %next_index : i32
  }

  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: int32 = 1

  // The following ops are skipped due to cancel.
  %x = hex.constant.i32 -1
  %ch1 = hex.new.chain
  hex.print.i32 %x, %ch1

  hex.return %repeat_result : i32
}
// CHECK: 'controlflow_repeat_cancel' returned <<error: Cancelled>>

// BEFExecutor will allocate an IndirectAsyncValue for this function's return
// value.
func @indirect_async_return(%c1: i32) -> i32 {
  %v2 = "hex.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "hex.async_add.i32"(%v2, %v2) : (i32, i32) -> i32
  // We can't return %v2 because it's an unavailable ConcreteAsyncValue (not an
  // IndirectAsyncValue).
  hex.return %v3 : i32
}

// Test an unused IndirectAsync return value.
// CHECK-LABEL: --- Running 'unused_indirect_async_return'
func @unused_indirect_async_return() {
  %c1 = hex.constant.i32 1
  // The returned IndirectAsyncValue is unused, but it has not yet been
  // forwarded. Setting this IndirectAsyncValue must count as a use, otherwise
  // the executor would immediately destroy the unused value.
  hex.call @indirect_async_return(%c1) : (i32) -> i32

  hex.return
}

// hex.cond tests.

func @identity(%x: i32) -> i32 {
  hex.return %x : i32
}

func @double(%x: i32) -> i32 {
  %y = hex.add.i32 %x, %x
  hex.return %y : i32
}

// CHECK-LABEL: --- Running 'if_test_with_func'
func @if_test_with_func() {
  %ch0 = hex.new.chain

  %a = hex.constant.i32 41

  %true = hex.constant.i1 1
  %true_res = hex.cond %true @identity @double (%a) : (i32) -> (i32)

  // CHECK-NEXT: int32 = 41
  %ch1 = hex.print.i32 %true_res, %ch0

  %false = hex.constant.i1 0
  %false_res = hex.cond %false @identity @double (%a) : (i32) -> (i32)

  // CHECK-NEXT: int32 = 82
  %ch2 = hex.print.i32 %false_res, %ch1

  hex.return
}
