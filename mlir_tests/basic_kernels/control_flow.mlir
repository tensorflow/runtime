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

// RUN: bef_executor %s.bef | FileCheck %s
// RUN: bef_executor -work_queue_type=mstd %s.bef | FileCheck %s

// This function is just a select: "cond ? v1 : v2", exercising tfrt.if with
// result values.
func @controlflow_if(%cond: i1, %v1: i32, %v2: i32) -> i32 {
  %res = tfrt.if %cond, %v1, %v2 : (i32, i32) -> (i32) {
    tfrt.return %v1 : i32
  } else {
    tfrt.return %v2 : i32
  }

  tfrt.return %res : i32
}

// CHECK-LABEL: --- Running 'if_test_0'
func @if_test_0() {
  %ch0 = tfrt.new.chain

  %a = tfrt.constant.i32 45
  %b = tfrt.constant.i32 129

  %true = tfrt.constant.i1 1

  // CHECK-NEXT: int32 = 45
  %x = tfrt.call @controlflow_if(%true, %a, %b) : (i1, i32, i32) -> (i32)
  %ch1 = tfrt.print.i32 %x, %ch0

  %false = tfrt.constant.i1 0

  // CHECK-NEXT: int32 = 129
  %y = tfrt.call @controlflow_if(%false, %a, %b) : (i1, i32, i32) -> (i32)
  %ch2 = tfrt.print.i32 %y, %ch1

  tfrt.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat2'
func @controlflow_repeat2() {
  %count = tfrt.constant.i32 4
  %ch0 = tfrt.new.chain

  %v0 = tfrt.constant.i32 1

  // This loop properly passes around the control chain to guarantee proper
  // sequencing of the print calls.
  //
  // The body of a repeat is a function call.  The "call" to it is passed the
  // %ch0 and %v0 values, the body of it receives them as %c1/%v1.
  %ch3:2 = tfrt.repeat.i32 %count, %ch0, %v0 : !tfrt.chain, i32 {
    %ch2 = tfrt.print.i32 %v0, %ch0
    %v2 = tfrt.add.i32 %v0, %v0
    tfrt.return %ch2, %v2 : !tfrt.chain, i32
  }

  // CHECK-NEXT: int32 = 1
  // CHECK-NEXT: int32 = 2
  // CHECK-NEXT: int32 = 4
  // CHECK-NEXT: int32 = 8

  %x = tfrt.constant.i32 -1

  // CHECK-NEXT: hello host executor!
  %ch4 = "tfrt_test.print_hello"(%ch3#0) : (!tfrt.chain) -> !tfrt.chain

  // CHECK-NEXT: int32 = 16
  tfrt.print.i32 %ch3#1, %ch4

  tfrt.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat5'
func @controlflow_repeat5() {
  %ch0 = tfrt.new.chain
  %count = tfrt.constant.i32 5
  %ch1 = tfrt.repeat.i32 %count, %ch0 : !tfrt.chain {
    %ch1 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain

    %x = tfrt.constant.i32 0
    %ch2 = tfrt.print.i32 %x, %ch1

    tfrt.return %ch2 : !tfrt.chain
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

  %x = tfrt.constant.i32 -1

  // CHECK-NEXT: int32 = -1
  tfrt.print.i32 %x, %ch1

  tfrt.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat_large'
func @controlflow_repeat_large() {
  %count = tfrt.constant.i32 1000
  %v0 = tfrt.constant.i32 42

  //TODO(xldrx): Fix the race condition when there are multiple worker threads.
  %sum = tfrt.repeat.i32 %count, %v0: i32 {
    %one = tfrt.constant.i32 1
    %v2 = "tfrt_test.async_add.i32"(%v0, %one) : (i32, i32) -> i32
    tfrt.return %v2: i32
  }

  %ch0 = tfrt.new.chain
  // CHECK-NEXT: int32 = 1042
  tfrt.print.i32 %sum, %ch0

  tfrt.return
}

// CHECK-LABEL: --- Running 'controlflow_repeat_cancel'
func @controlflow_repeat_cancel() -> i32 {
  %ch0 = tfrt.new.chain
  %count = tfrt.constant.i32 5
  %index = tfrt.constant.i32 0

  %ch1, %repeat_result = tfrt.repeat.i32 %count, %ch0, %index : !tfrt.chain, i32 {
    %ch = tfrt.print.i32 %index, %ch0

    %one = tfrt.constant.i32 1
    %cond = "tfrt.lessequal.i32"(%one, %index, %ch) : (i32, i32, !tfrt.chain) -> (i1)

    // Cancel when the loop index reaches 1.
    %ch1, %x = tfrt.if %cond, %ch, %one : (!tfrt.chain, i32) -> (!tfrt.chain, i32) {
      %x, %ch1 = "tfrt_test.cancel"(%ch) : (!tfrt.chain) -> (i32, !tfrt.chain)
      tfrt.return %ch1, %x : !tfrt.chain, i32
    } else {
      tfrt.return %ch, %one : !tfrt.chain, i32
    }

    %next_index = tfrt.add.i32 %index, %x
    tfrt.return %ch1, %next_index : !tfrt.chain, i32
  }

  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: int32 = 1

  // The following ops are skipped due to cancel.
  %x = tfrt.constant.i32 -1
  tfrt.print.i32 %x, %ch1

  tfrt.return %repeat_result : i32
}
// CHECK: 'controlflow_repeat_cancel' returned <<error: Cancelled>>

// BEFExecutor will allocate an IndirectAsyncValue for this function's return
// value.
func @indirect_async_return(%c1: i32) -> i32 {
  %v2 = "tfrt_test.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "tfrt_test.async_add.i32"(%v2, %v2) : (i32, i32) -> i32
  // We can't return %v2 because it's an unavailable ConcreteAsyncValue (not an
  // IndirectAsyncValue).
  tfrt.return %v3 : i32
}

// Test an unused IndirectAsync return value.
// CHECK-LABEL: --- Running 'unused_indirect_async_return'
func @unused_indirect_async_return() {
  %c1 = tfrt.constant.i32 1
  // The returned IndirectAsyncValue is unused, but it has not yet been
  // forwarded. Setting this IndirectAsyncValue must count as a use, otherwise
  // the executor would immediately destroy the unused value.
  tfrt.call @indirect_async_return(%c1) : (i32) -> i32

  tfrt.return
}

// tfrt.cond tests.

func @identity(%x: i32) -> i32 {
  tfrt.return %x : i32
}

func @double(%x: i32) -> i32 {
  %y = tfrt.add.i32 %x, %x
  tfrt.return %y : i32
}

// CHECK-LABEL: --- Running 'if_test_with_func'
func @if_test_with_func() {
  %ch0 = tfrt.new.chain

  %a = tfrt.constant.i32 41

  %true = tfrt.constant.i1 1
  %true_res = tfrt.cond %true @identity @double (%a) : (i32) -> (i32)

  // CHECK-NEXT: int32 = 41
  %ch1 = tfrt.print.i32 %true_res, %ch0

  %false = tfrt.constant.i1 0
  %false_res = tfrt.cond %false @identity @double (%a) : (i32) -> (i32)

  // CHECK-NEXT: int32 = 82
  %ch2 = tfrt.print.i32 %false_res, %ch1

  tfrt.return
}

func @branch0(%ch: !tfrt.chain, %arg: i32) -> (!tfrt.chain, i32) {
  %one = tfrt.constant.i32 2
  %res = tfrt.add.i32 %arg, %one
  tfrt.return %ch, %res : !tfrt.chain, i32
}

func @branch1(%ch: !tfrt.chain, %arg: i32) -> (!tfrt.chain, i32) {
  %two = tfrt.constant.i32 3
  %res = tfrt.add.i32 %arg, %two
  tfrt.return %ch, %res : !tfrt.chain, i32
}

// tfrt.case tests.
// CHECK-LABEL: --- Running 'tfrt_case_test'
func @tfrt_case_test() {
  %ch0 = tfrt.new.chain

  %branch_index = tfrt.constant.i32 0

  %arg = tfrt.constant.i32 40

  %ch1, %res = tfrt.case %branch_index [@branch0, @branch1] (%ch0, %arg) :  (i32) -> (i32)

  // CHECK: int32 = 42
  %ch2 = tfrt.print.i32 %res, %ch1

  %branch_index1 = tfrt.constant.i32 1

  %ch3, %res1 = tfrt.case %branch_index1 [@branch0, @branch1] (%ch2, %arg) :  (i32) -> (i32)

  // CHECK: int32 = 43
  %ch4 = tfrt.print.i32 %res1, %ch3

  tfrt.return
}

func @tfrt_while_body(%ch: !tfrt.chain, %iteration: i32, %arg: i32) -> (!tfrt.chain, i32, i32, i1) {
  %one = tfrt.constant.i32 1
  %five = tfrt.constant.i32 5
  %next_iteration = tfrt.add.i32 %iteration, %one
  %next_arg = tfrt.add.i32 %arg, %five
  %next_cond = "tfrt.lessequal.i32"(%next_iteration, %five) : (i32, i32) -> (i1)

  tfrt.return %ch, %next_iteration, %next_arg, %next_cond : !tfrt.chain, i32, i32, i1
}

// CHECK-LABEL: --- Running 'tfrt_inline_while_test'
func @tfrt_inline_while_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %cond = tfrt.constant.i1 true
  %iteration = tfrt.constant.i32 0
  %arg = tfrt.constant.i32 0

  %ch1, %final_iteration, %final_arg = tfrt.while %cond @tfrt_while_body(%ch0, %iteration, %arg) parallel_iterations(1) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)

  // CHECK: int32 = 30
  %ch2 = tfrt.print.i32 %final_arg, %ch1
  // CHECK: int32 = 6
  %ch3 = tfrt.print.i32 %final_iteration, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'tfrt_parallel_while_test'
func @tfrt_parallel_while_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %cond = tfrt.constant.i1 true
  %iteration = tfrt.constant.i32 0
  %arg = tfrt.constant.i32 0

  %ch1, %final_iteration, %final_arg = tfrt.while %cond @tfrt_while_body(%ch0, %iteration, %arg) parallel_iterations(4) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)

  // CHECK: int32 = 30
  %ch2 = tfrt.print.i32 %final_arg, %ch1
  // CHECK: int32 = 6
  %ch3 = tfrt.print.i32 %final_iteration, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

func @tfrt_while_error_body(%ch: !tfrt.chain, %iteration: i32, %arg: i32) -> (!tfrt.chain, i32, i32, i1) {
  %one = tfrt.constant.i32 1
  %five = tfrt.constant.i32 5
  %next_iteration = tfrt.add.i32 %iteration, %one
  %next_arg = tfrt.add.i32 %arg, %five
  %x, %ch1 = "tfrt_test.cancel"(%ch) : (!tfrt.chain) -> (i32, !tfrt.chain)
  %next_cond = "tfrt.lessequal.i32"(%x, %five) : (i32, i32) -> (i1)

  tfrt.return %ch1, %next_iteration, %next_arg, %next_cond : !tfrt.chain, i32, i32, i1
}

// CHECK-LABEL: --- Running 'tfrt_while_error_test'
func @tfrt_while_error_test() -> (!tfrt.chain, i32, i32) {
  %ch0 = tfrt.new.chain

  %cond = tfrt.constant.i1 true
  %iteration = tfrt.constant.i32 0
  %arg = tfrt.constant.i32 0

  %ch1, %final_iteration, %final_arg = tfrt.while %cond @tfrt_while_error_body(%ch0, %iteration, %arg) parallel_iterations(1) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)

  // CHECK: 'tfrt_while_error_test' returned <<error: Cancelled>>,<<error: Cancelled>>,<<error: Cancelled>>
  tfrt.return %ch1, %final_iteration, %final_arg : !tfrt.chain, i32, i32
}

// CHECK-LABEL: --- Running 'tfrt_parallel_while_error_test'
func @tfrt_parallel_while_error_test() -> (!tfrt.chain, i32, i32) {
  %ch0 = tfrt.new.chain

  %cond = tfrt.constant.i1 true
  %iteration = tfrt.constant.i32 0
  %arg = tfrt.constant.i32 0

  %ch1, %final_iteration, %final_arg = tfrt.while %cond @tfrt_while_error_body(%ch0, %iteration, %arg) parallel_iterations(4) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)

  // CHECK: 'tfrt_parallel_while_error_test' returned <<error: Cancelled>>,<<error: Cancelled>>,<<error: Cancelled>>
  tfrt.return %ch1, %final_iteration, %final_arg : !tfrt.chain, i32, i32
}

func @tfrt_once_function(%arg : i32) -> i32 {
  %one = tfrt.constant.i32 1
  %result = tfrt.add.i32 %arg, %one
  tfrt.return %result : i32
}

// CHECK-LABEL: --- Running 'tfrt_once_test'
func @tfrt_once_test() {
  %0 = tfrt.constant.i32 0
  %1 = tfrt.once @tfrt_once_function(%0) : (i32) -> (i32)
  %2 = tfrt.once @tfrt_once_function(%1) : (i32) -> (i32)

  %ch0 = tfrt.new.chain
  // CHECK-NEXT: int32 = 1
  %ch1 = tfrt.print.i32 %1, %ch0
  // CHECK-NEXT: int32 = 1
  %ch2 = tfrt.print.i32 %2, %ch1

  tfrt.return
}

// CHECK-LABEL: --- Running 'tfrt_merge_chain_test'
func @tfrt_merge_chain_test() -> !tfrt.chain {
  %v = tfrt.constant.i32 0
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.merge.chains %v, %ch0 : i32, !tfrt.chain

  // CHECK: 'tfrt_merge_chain_test' returned !tfrt.chain value
  tfrt.return %ch1 : !tfrt.chain
}
