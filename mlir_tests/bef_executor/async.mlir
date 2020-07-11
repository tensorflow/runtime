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

// RUN: bef_executor -work_queue_type=s $(bef_name %s) | FileCheck %s --dump-input=fail

// NOTE: This test is intentionally not using chains to sequence side effecting
// kernels according to common sense.  This is to make sure the executor is
// running them in an expected order based on its internal implementation
// details. Note that this test requires the use of single-threaded work queue.

// CHECK-LABEL: --- Running 'async_add'
func @async_add() {
  %ch0 = hex.new.chain

  %x = hex.constant.i32 42
  %c1 = hex.constant.i32 1

  // CHECK: int32 = 42
  hex.print.i32 %x, %ch0

  %y = tfrt_test.do.async %x, %c1 : (i32, i32) -> (i32) {
    %y_res = hex.add.i32 %x, %c1
    hex.return %y_res : i32
  }

  hex.print.i32 %y, %ch0

  // CHECK: int32 = 42
  hex.print.i32 %x, %ch0

  // 43 is printed after async_add.i32 completes.
  // CHECK: int32 = 43
  hex.return
}

// CHECK-LABEL: -- Running 'async_repeat2'
func @async_repeat2() {
  %count = hex.constant.i32 2

  hex.repeat.i32 %count {
    tfrt_test.do.async : () -> () {
      %ch0 = hex.new.chain
      "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain

      %x = hex.constant.i32 0
      hex.print.i32 %x, %ch0
      hex.return
    }
    hex.return
  }

  %ch3 = hex.new.chain

  %x = hex.constant.i32 -1
  // CHECK-NEXT: int32 = -1
  hex.print.i32 %x, %ch3

  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  hex.return
}

// CHECK-LABEL: --- Not running 'call_async_add.i32' because it has arguments
func @call_async_add.i32(%x: i32, %y: i32) -> i32 {
  %a = "hex.async_add.i32"(%x, %y) : (i32, i32) -> i32
  %b = "hex.async_add.i32"(%a, %y) : (i32, i32) -> i32
  hex.return %b : i32
}

// CHECK-LABEL: --- Running 'async_add.i32_caller'
func @async_add.i32_caller() {
  %x = hex.constant.i32 42
  %c1 = hex.constant.i32 1
  %ch = hex.new.chain

  // CHECK: int32 = 42
  hex.print.i32 %x, %ch

  %y = hex.call @call_async_add.i32(%x, %c1) : (i32, i32) -> i32
  hex.print.i32 %y, %ch

  // CHECK: int32 = 42
  hex.print.i32 %x, %ch

  // 44 is printed after call_async_add.i32 completes.
  // CHECK: int32 = 44
  hex.return
}

// CHECK-LABEL: --- Running 'test_async_print_result'
func @test_async_print_result() -> i32 {
  %c1 = hex.constant.i32 1
  %c42 = hex.constant.i32 42
  %a = "hex.async_add.i32"(%c1, %c42) : (i32, i32) -> i32

  hex.return %a : i32
}
// CHECK: 'test_async_print_result' returned 43

// CHECK-LABEL: --- Running 'test_async_copy'
func @test_async_copy() -> i32 {
  %c42 = hex.constant.i32 42
  %copy = "hex.async_copy.i32"(%c42) : (i32) -> i32

  hex.return %copy : i32
}
// CHECK: 'test_async_copy' returned 42

// CHECK-LABEL: --- Running 'test_async_copy.with_delay'
func @test_async_copy.with_delay() -> i32 {
  %c42 = hex.constant.i32 42
  %copy = "hex.async_copy.with_delay.i32"(%c42) : (i32) -> i32

  hex.return %copy : i32
}
// CHECK: 'test_async_copy.with_delay' returned 42

// CHECK-LABEL: --- Running 'test_async_copy_2'
func @test_async_copy_2() -> i32 {
  %c43 = hex.constant.i32 43
  %copy = "hex.async_copy_2.i32"(%c43) : (i32) -> i32

  hex.return %copy : i32
}
// CHECK: 'test_async_copy_2' returned 43
