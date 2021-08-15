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

// RUN: bef_executor_lite -work_queue_type=s %s.bef | FileCheck %s

// NOTE: This test is intentionally not using chains to sequence side effecting
// kernels according to common sense.  This is to make sure the executor is
// running them in an expected order based on its internal implementation
// details. Note that this test requires the use of single-threaded work queue.

// CHECK-LABEL: --- Running 'async_add'
func @async_add() {
  %ch0 = tfrt.new.chain

  %x = tfrt.constant.i32 42
  %c1 = tfrt.constant.i32 1

  // CHECK: int32 = 42
  %ch1 = tfrt.print.i32 %x, %ch0

  %y = tfrt_test.do.async %x, %c1 : (i32, i32) -> (i32) {
    %y_res = tfrt.add.i32 %x, %c1
    tfrt.return %y_res : i32
  }

  // CHECK-NEXT: int32 = 42
  // CHECK-NEXT: int32 = 43
  %ch2 = tfrt.print.i32 %x, %ch1
  %ch3 = tfrt.print.i32 %y, %ch2

  tfrt.return
}

// CHECK-LABEL: -- Running 'async_repeat2'
func @async_repeat2() {
  %count = tfrt.constant.i32 2
  %ch = tfrt.new.chain

  %ch3 = tfrt.repeat.i32 %count, %ch : !tfrt.chain {
    %async_ch = tfrt_test.do.async %ch : (!tfrt.chain) -> (!tfrt.chain) {
      %ch1 = "tfrt_test.print_hello"(%ch) : (!tfrt.chain) -> !tfrt.chain

      %x = tfrt.constant.i32 0
      %ch2 = tfrt.print.i32 %x, %ch1
      tfrt.return %ch2 : !tfrt.chain
    }
    tfrt.return %async_ch : !tfrt.chain
  }

  %x = tfrt.constant.i32 -1
  tfrt.print.i32 %x, %ch3

  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello host executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: int32 = -1
  tfrt.return
}

// CHECK-LABEL: --- Not running 'call_async_add.i32' because it has arguments
func @call_async_add.i32(%x: i32, %y: i32) -> i32 {
  %a = "tfrt_test.async_add.i32"(%x, %y) : (i32, i32) -> i32
  %b = "tfrt_test.async_add.i32"(%a, %y) : (i32, i32) -> i32
  tfrt.return %b : i32
}

// CHECK-LABEL: --- Running 'async_add.i32_caller'
func @async_add.i32_caller() {
  %ch = tfrt.new.chain
  %x = tfrt.constant.i32 42
  %c1 = tfrt.constant.i32 1

  // CHECK: int32 = 42
  tfrt.print.i32 %x, %ch

  %y = tfrt.call @call_async_add.i32(%x, %c1) : (i32, i32) -> i32
  tfrt.print.i32 %y, %ch

  // CHECK: int32 = 42
  tfrt.print.i32 %x, %ch

  // 44 is printed after call_async_add.i32 completes.
  // CHECK: int32 = 44
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_async_print_result'
func @test_async_print_result() -> i32 {
  %c1 = tfrt.constant.i32 1
  %c42 = tfrt.constant.i32 42
  %a = "tfrt_test.async_add.i32"(%c1, %c42) : (i32, i32) -> i32

  tfrt.return %a : i32
}
// CHECK: 'test_async_print_result' returned 43

// CHECK-LABEL: --- Running 'test_async_copy'
func @test_async_copy() -> i32 {
  %c42 = tfrt.constant.i32 42
  %copy = "tfrt_test.async_copy.i32"(%c42) : (i32) -> i32

  tfrt.return %copy : i32
}
// CHECK: 'test_async_copy' returned 42

// CHECK-LABEL: --- Running 'test_async_copy.with_delay'
func @test_async_copy.with_delay() -> i32 {
  %c42 = tfrt.constant.i32 42
  %copy = "tfrt_test.async_copy.with_delay.i32"(%c42) : (i32) -> i32

  tfrt.return %copy : i32
}
// CHECK: 'test_async_copy.with_delay' returned 42

// CHECK-LABEL: --- Running 'test_async_copy_2'
func @test_async_copy_2() -> i32 {
  %c43 = tfrt.constant.i32 43
  %copy = "tfrt_test.async_copy_2.i32"(%c43) : (i32) -> i32

  tfrt.return %copy : i32
}
// CHECK: 'test_async_copy_2' returned 43

// CHECK-LABEL: --- Running 'test_as_chain'
func @test_as_chain() -> !tfrt.chain {
  %c43 = tfrt.constant.i32 43
  %ch = "tfrt_test.as_chain"(%c43) : (i32) -> !tfrt.chain

  // CHECK: 'test_as_chain' returned !tfrt.chain
  tfrt.return %ch : !tfrt.chain
}
