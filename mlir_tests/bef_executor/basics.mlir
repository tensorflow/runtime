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

// CHECK: --- Running 'print_test'
func @print_test() {
  // CHECK: hello host executor!
  %ch0 = tfrt.new.chain
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return
}

// CHECK-LABEL: --- Running 'three_print_error'
func @three_print_error() {
  %ch0 = tfrt.new.chain

  // CHECK: hello host executor!
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain

  // CHECK: hello host executor!
  %ch2 = "tfrt_test.print_hello"(%ch1) : (!tfrt.chain) -> !tfrt.chain

  // CHECK: hello host executor!
  %ch3 = "tfrt_test.print_hello"(%ch2) : (!tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'basic.i1'
func @basic.i1() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.i1 0
  // CHECK: int1 = 0
  %ch1 = tfrt.print.i1 %zero, %ch0

  %one = tfrt.constant.i1 1
  // CHECK: int1 = 1
  %ch2 = tfrt.print.i1 %one, %ch1

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic.i32'
func @basic.i32() -> !tfrt.chain {
  %x = tfrt.constant.i32 42

  %ch0 = tfrt.new.chain

  // CHECK: int32 = 42
  %ch1 = tfrt.print.i32 %x, %ch0

  %c1 = tfrt.constant.i32 1
  // CHECK: int32 = 1
  %ch2 = tfrt.print.i32 %c1, %ch1

  %y = tfrt.add.i32 %x, %c1
  // CHECK: int32 = 43
  %ch3 = tfrt.print.i32 %y, %ch2

  %quot, %rem = tfrt.div.i32 %y, %x
  // CHECK: int32 = 1
  %ch4 = tfrt.print.i32 %quot, %ch3
  // CHECK: int32 = 1
  %ch5 = tfrt.print.i32 %rem, %ch4

  %y1, %y2, %y3 = tfrt_test.count3.i32 %y
  // CHECK: int32 = 44
  %ch6 = tfrt.print.i32 %y1, %ch5

  // CHECK: int32 = 45
  %ch7 = tfrt.print.i32 %y2, %ch6

  // CHECK: int32 = 46
  %ch8 = tfrt.print.i32 %y3, %ch7

  %y4 = "tfrt_test.copy.with_delay.i32"(%y3) : (i32) -> i32
  // CHECK: int32 = 46
  %ch9 = tfrt.print.i32 %y4, %ch8

  tfrt.return %ch9 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'divide_by_zero.i32'
func @divide_by_zero.i32() -> i32 {
  %zero = tfrt.constant.i32 0
  %x = tfrt.constant.i32 42

  // expected-error @+1 {{runtime error: Divide by zero}}
  %quot, %rem = tfrt.div.i32 %x, %zero

  tfrt.return %quot : i32
}

// CHECK-LABEL: --- Running 'basic.i64'
func @basic.i64() -> !tfrt.chain {
  %x = tfrt.constant.i64 42

  %ch0 = tfrt.new.chain

  // CHECK: int64 = 42
  %ch1 = tfrt.print.i64 %x, %ch0

  %c1 = tfrt.constant.i64 1
  // CHECK: int64 = 1
  %ch2 = tfrt.print.i64 %c1, %ch1

  %y = tfrt.add.i64 %x, %c1
  // CHECK: int64 = 43
  %ch3 = tfrt.print.i64 %y, %ch2

  %z = "tfrt_test.copy.with_delay.i64"(%y) : (i64) -> i64
  // CHECK: int64 = 43
  %ch4 = tfrt.print.i64 %z, %ch3

  tfrt.return %ch4 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic.f32'
func @basic.f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  // CHECK: f32 = 0.000000
  %ch1 = tfrt.print.f32 %zero, %ch0

  %one = tfrt.constant.f32 1.0
  // CHECK: f32 = 1.000000
  %ch2 = tfrt.print.f32 %one, %ch1

  %six = tfrt.constant.f32 6.0
  %seven = tfrt.add.f32 %one, %six
  // CHECK: f32 = 7.000000
  %ch3 = tfrt.print.f32 %seven, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic.f64'
func @basic.f64() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f64 0.0
  // CHECK: f64 = 0.000000
  %ch1 = tfrt.print.f64 %zero, %ch0

  %one = tfrt.constant.f64 1.0
  // CHECK: f64 = 1.000000
  %ch2 = tfrt.print.f64 %one, %ch1

  %six = tfrt.constant.f64 6.0
  %seven = tfrt.add.f64 %one, %six
  // CHECK: f64 = 7.000000
  %ch3 = tfrt.print.f64 %seven, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic_strings'
func @basic_strings() {
  %x = "tfrt_test.get_string"() { value = "bark" } : () -> !tfrt.string
  %y = "tfrt_test.get_string"() { value = "rarf" } : () -> !tfrt.string
  %ch0 = tfrt.new.chain

  // CHECK: string = bark
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  %z = "tfrt_test.append_string"(%x, %y) : (!tfrt.string, !tfrt.string) -> !tfrt.string

  // CHECK: string = barkrarf
  "tfrt_test.print_string"(%z, %ch1) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// CHECK-LABEL: --- Not running 'call_print.i32' because it has arguments
func @call_print.i32(%x: i32) -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.print.i32 %x, %ch0
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'add_one' because it has arguments
func @add_one(%x: i32) -> i32 {
  %c1 = tfrt.constant.i32 1
  %y = tfrt.add.i32 %x, %c1
  tfrt.return %y : i32
}

// CHECK-LABEL: --- Running 'caller'
func @caller() -> !tfrt.chain {
  %c1 = tfrt.constant.i32 1

  // CHECK-NEXT: int32 = 1
  %ch0 = tfrt.call @call_print.i32(%c1) : (i32) -> (!tfrt.chain)

  %x = tfrt.call @add_one(%c1) : (i32) -> i32

  // CHECK-NEXT: int32 = 2
  %ch1 = tfrt.print.i32 %x, %ch0

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_error_result'
func @test_error_result() -> i32 {
  %x = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_error_result_concrete_async_success'
func @test_error_result_concrete_async_success() -> i32 {
  %in = tfrt.constant.i32 0
  %x = "tfrt_test.report_error_concrete_async"(%in) : (i32) -> i32
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result_concrete_async_success' returned 0

// CHECK-LABEL: --- Running 'test_error_result_concrete_async_fail'
func @test_error_result_concrete_async_fail() -> i32 {
  %in = tfrt.constant.i32 1
  %x = "tfrt_test.report_error_concrete_async"(%in) : (i32) -> i32 // expected-error {{something bad happened asynchronously}}

  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result_concrete_async_fail' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_error_result_indirect_async_success'
func @test_error_result_indirect_async_success() -> i32 {
  %in = tfrt.constant.i32 0
  %x = "tfrt_test.report_error_indirect_async"(%in) : (i32) -> i32
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result_indirect_async_success' returned 0

// CHECK-LABEL: --- Running 'test_error_result_indirect_async_fail'
func @test_error_result_indirect_async_fail() -> i32 {
  %in = tfrt.constant.i32 1
  %x = "tfrt_test.report_error_indirect_async"(%in) : (i32) -> i32 // expected-error {{something bad happened asynchronously}}

  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result_indirect_async_fail' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_ignore_error'
func @test_ignore_error() -> i32 {
  "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}
  %x = tfrt.constant.i32 1
  tfrt.return %x : i32
}
// CHECK: 'test_ignore_error' returned 1

// CHECK-LABEL: --- Running 'test_partial_fail'
func @test_partial_fail() -> !tfrt.chain {
  // Sets %x to 1 and %y to an error.
  %x, %y = "tfrt_test.partial_fail"() : () -> (i32, i32) // expected-error {{something bad happened}}

  %ch0 = tfrt.new.chain
  // CHECK: int32 = 1
  %ch1 = tfrt.print.i32 %x, %ch0

  // CHECK-NOT: int32
  %ch2 = tfrt.print.i32 %y, %ch1

  tfrt.return %ch2 : !tfrt.chain
}
// CHECK-NEXT: 'test_partial_fail' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_cancel'
func @test_cancel() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %x, %ch1 = "tfrt_test.cancel"(%ch0) : (!tfrt.chain) -> (i32, !tfrt.chain)

  // The following kernels are skipped due to cancellation.
  // CHECK-NOT: int32 = 0
  %ch2 = tfrt.print.i32 %x, %ch1

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-NEXT: 'test_cancel' returned <<error: Cancelled>>

// CHECK-LABEL: --- Running 'test_async_value_get'
func @test_async_value_get() -> () {
  %x = "tfrt_test.async_value_get"() : () -> !tfrt.string

  %ch0 = tfrt.new.chain
  // CHECK: string = base1:base2:final_class:3
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_async_value_ref'
func @test_async_value_ref() -> () {
  %x = "tfrt_test.async_value_ref"() : () -> !tfrt.string

  %ch0 = tfrt.new.chain
  // CHECK: string = available:2;unavailable;(3);available:3;unavailable;available:4
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_logging'
func @test_logging() {
  %x = "tfrt_test.get_string"() { value = "hello world" } : () -> !tfrt.string
  %c1 = tfrt.new.chain
  // CHECK: from TFRT_LOG(INFO): hello world
  // CHECK: from TFRT_LOG(WARNING): hello world
  // CHECK: from TFRT_LOG(ERROR): hello world
  // CHECK: from TFRT_LOG_IF(INFO, true): hello world
  // CHECK: from TFRT_LOG_IF(WARNING, true): hello world
  // CHECK: from TFRT_LOG_IF(ERROR, true): hello world
  %c2 = "tfrt_test.logging" (%x, %c1) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_explicit_resource_management'
func @test_explicit_resource_management() -> !tfrt.chain {
  %c0 = tfrt.new.chain

  // CHECK-NEXT: Allocated TestResource
  %res, %c1 = "tfrt_test.allocate_resource" (%c0) : (!tfrt.chain) -> (!tfrt.test.resource, !tfrt.chain)

  // test.deallocate_resource explicitly destroys TestResource, so deallocation
  // happens before test.deallocate_resource completes.
  //
  // CHECK-NEXT: Destroyed TestResource
  // CHECK-NEXT: test.deallocate_resource done
  %c2 = "tfrt_test.deallocate_resource" (%res, %c1) : (!tfrt.test.resource, !tfrt.chain) -> (!tfrt.chain)
  tfrt.return %c2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_explicit_resource_management_with_cancellation'
func @test_explicit_resource_management_with_cancellation() -> !tfrt.chain {
  %c0 = tfrt.new.chain
  // CHECK-NEXT: Allocated TestResource
  %res, %c1 = "tfrt_test.allocate_resource" (%c0) : (!tfrt.chain) -> (!tfrt.test.resource, !tfrt.chain)

  %x, %c2 = "tfrt_test.cancel"(%c1) : (!tfrt.chain) -> (i32, !tfrt.chain)

  // Cancellation skips test.deallocate_resource, but TestResource is still
  // destroyed.
  //
  // CHECK-NEXT: Destroyed TestResource
  %c3 = "tfrt_test.deallocate_resource" (%res, %c2) : (!tfrt.test.resource, !tfrt.chain) -> (!tfrt.chain)
  tfrt.return %c3 : !tfrt.chain
}
// CHECK-NEXT: 'test_explicit_resource_management_with_cancellation' returned <<error: Cancelled>>

// CHECK-LABEL: --- Running 'test_shared_context'
func @test_shared_context() {
  %ch0 = tfrt.new.chain

  // use the sample shared context twice to verify only one instance is created.
  %name1, %ch1 = "tfrt_test.use_sample_shared_context"(%ch0) : (!tfrt.chain) -> (!tfrt.string, !tfrt.chain)
  %name2, %ch2 = "tfrt_test.use_sample_shared_context"(%ch0) : (!tfrt.chain) -> (!tfrt.string, !tfrt.chain)

  // CHECK: string = sample_shared_context0
  %ch3 = "tfrt_test.print_string"(%name1, %ch1) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: string = sample_shared_context0
  %ch4 = "tfrt_test.print_string"(%name2, %ch2) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// CHECK-LABEL: --- Running 'return_multi'
func @return_multi() -> (i32, i32, i32, i32) {
  %a = tfrt.constant.i32 1
  %b = tfrt.constant.i32 2

  // CHECK: 'return_multi' returned 1,1,1,2
  tfrt.return %a, %a, %a, %b : i32, i32, i32, i32
}

func @nested_array() {
  %ch0 = tfrt.new.chain

  %a, %b, %c, %d = "tfrt_test.flat"() { value = [["string", [1 : i32, 0 : i32]], [1.0 : f32]]} : () -> (!tfrt.string, i32, i32, f32)

  // CHECK: string = string
  %ch1 = "tfrt_test.print_string"(%a, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)
  // CHECK: int32 = 1
  %ch2 = tfrt.print.i32 %b, %ch1
  // CHECK: int32 = 0
  %ch3 = tfrt.print.i32 %c, %ch2
  // CHECK: f32 = 1.000000
  %ch4 = tfrt.print.f32 %d, %ch3

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_error_result_async'
func @test_error_result_async() -> i32 {
  %x = "tfrt_test.report_error_async"() : () -> i32 // expected-error {{something bad happened asynchronously}}
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result_async' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_error_result_async_unused'
func @test_error_result_async_unused() -> i32 {
  %x = tfrt.constant.i32 123
  %y = "tfrt_test.report_error_async"() : () -> i32  // expected-error {{something bad happened asynchronously}}
  // CHECK: 'test_error_result_async_unused' returned 123
  tfrt.return %x : i32
}

// CHECK-LABEL: --- Running 'test_identity'
func @test_identity() {
  %ch = tfrt.new.chain
  %x = tfrt.constant.i32 123
  %y = "tfrt_test.identity"(%ch, %x) : (!tfrt.chain, i32) -> (i32)
  // CHECK: int32 = 123
  tfrt.print.i32 %y, %ch
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_multi_arg_result'
func @test_multi_arg_result() -> !tfrt.chain {
  %x = tfrt.constant.i32 1
  %y = tfrt.constant.i32 2
  %z = tfrt.constant.i32 3

  %x1, %y1, %z1 = "tfrt_test.multi_arg_result"(%x, %y, %z) : (i32, i32, i32) -> (i32, i32, i32)

  // CHECK: int32 = 1
  // CHECK: int32 = 2
  // CHECK: int32 = 3
  %ch = tfrt.new.chain
  %ch1 = tfrt.print.i32 %x1, %ch
  %ch2 = tfrt.print.i32 %y1, %ch1
  %ch3 = tfrt.print.i32 %z1, %ch2
  tfrt.return %ch3 : !tfrt.chain
}
