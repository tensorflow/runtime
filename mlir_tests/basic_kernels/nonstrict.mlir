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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail

// NOTE: This test is intentionally not using chains to sequence side effecting
// kernels according to common sense.  This is to make sure the executor is
// running them in an expected order based on its internal implementation
// details.

// CHECK-LABEL: --- Not running 'callee' because it has arguments

// This function prints the values as they arrive.
func @callee(%x: i32, %y: i32, %z: i32) {
  %ch0 = hex.new.chain

  hex.print.i32 %x, %ch0
  hex.print.i32 %y, %ch0
  hex.print.i32 %z, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'call_non_strict'
func @call_non_strict() {
  %c1 = hex.constant.i32 1

  %v2 = "hex.async_add.i32"(%c1, %c1) : (i32, i32) -> i32

  %v3 = "hex.async_add.i32"(%c1, %v2) : (i32, i32) -> i32

  // CHECK-NEXT: int32 = 1
  hex.call @callee(%c1, %v2, %v3) : (i32, i32, i32) -> ()

  %c5 = hex.constant.i32 5

  %ch0 = hex.new.chain

  // CHECK-NEXT: int32 = 5
  hex.print.i32 %c5, %ch0

  // CHECK-NEXT: int32 = 2
  // CHECK-NEXT: int32 = 3

  hex.return
}

// CHECK-LABEL: --- Running 'if_non_strict'
func @if_non_strict() {
  // Note: these prints are intentionally unsequenced so we can test which
  // async values get resolved first.
  %ch0 = hex.new.chain

  %true = hex.constant.i1 1
  %c1 = hex.constant.i32 1

  // v2 gets resolved later.
  %v2 = "hex.async_add.i32"(%c1, %c1) : (i32, i32) -> i32

  // This is a *non-strict* if.
  hex.if %true, %c1, %v2 : (i32, i32) -> () {
    // Note: these prints are intentionally unsequenced so we can test which
    // async values get resolved first.
    %ch3 = hex.new.chain
    // We are non-strict, so c1 is ready now and v2 will be resolved later.
    hex.print.i32 %c1, %ch3
    hex.print.i32 %v2, %ch3
    hex.return
  }

  // Because the if is non-strict, it immediately invokes its body, which
  // immediately prints c1. Printing v2 is delayed until v2 is resolved by
  // HostContext::Quiesce().

  // CHECK-NEXT: int32 = 1
  // CHECK-NEXT: hello host executor!
  %ch2 = "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain
  // CHECK-NEXT: int32 = 2

  hex.return
}

// Test hex.if with a deferred condition.
// CHECK-LABEL: --- Running 'if_non_strict_condition'
func @if_non_strict_condition() {
  %ch0 = hex.new.chain

  // %true gets resolved later.
  %true = "hex.async_constant.i1"() { value = 1 : i1 } : () -> i1

  %c1 = hex.constant.i32 1
  %c2 = hex.constant.i32 2

  // This is a *non-strict* if where the condition is resolved later.
  // This hex.if returns an unused value.
  hex.if %true, %c1, %c2 : (i32, i32) -> (i32) {
    %ch3 = hex.new.chain
    %ch4 = hex.print.i32 %c1, %ch3
    hex.print.i32 %c2, %ch4
    hex.return %c1 : i32
  } else {
    hex.return %c1 : i32
  }

  // CHECK-NEXT: hello host executor!
  %ch2 = "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain

  // Because the if is non-strict, and its condition is not ready, the true
  // branch is not invoked until the condition is resolved by
  // HostContext::Quiesce().

  // CHECK-NEXT: int32 = 1
  // CHECK-NEXT: int32 = 2
  hex.return
}

// Test hex.repeat with a deferred count.
// CHECK-LABEL: --- Running 'repeat_non_strict_count'
func @repeat_non_strict_count() {
  // count gets resolved later.
  %count = "hex.async_constant.i32"() { value = 2 : i32 } : () -> i32

  %ch0 = hex.new.chain
  %v0 = hex.constant.i32 1

  // This is a *non-strict* repeat where the count is resolved later.
  // This hex.repeat.i32 returns unused values.
  hex.repeat.i32 %count, %ch0, %v0 : !hex.chain, i32 {
    %ch2 = hex.print.i32 %v0, %ch0
    hex.return %ch2, %v0 : !hex.chain, i32
  }

  %x = hex.constant.i32 -1

  // CHECK-NEXT: hello host executor!
  "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain

  // Because the repeat is non-strict, and its count is not ready, the body is
  // not invoked until the count is resolved by HostContext::Quiesce().

  // CHECK-NEXT: int32 = 1
  // CHECK-NEXT: int32 = 1
  hex.return
}

func @return_first_arg(%x: i32, %y: i32) -> i32 {
  hex.return %x : i32
}

// Test a nonstrict call with an unused argument.
// CHECK-LABEL: --- Running 'unused_nonstrict_arg'
func @unused_nonstrict_arg() {
  %c1 = hex.constant.i32 1

  %v2 = "hex.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  %v3 = "hex.async_add.i32"(%v2, %v2) : (i32, i32) -> i32

  // BEFExecutor will allocate an IndirectAsyncValue for %v3, but %v3 is unused.
  %x = hex.call @return_first_arg(%c1, %v3) : (i32, i32) -> i32

  // At this point %v3 is unused, but has not yet been forwarded. Setting %v3
  // must count as a use, otherwise the executor would immediately destroy %v3.

  %ch0 = hex.new.chain
  // CHECK-NEXT: int32 = 1
  hex.print.i32 %x, %ch0

  hex.return
}
