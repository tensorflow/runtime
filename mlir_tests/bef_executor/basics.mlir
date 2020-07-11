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

// RUN: bef_executor $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail

// CHECK: --- Running 'print_test'
func @print_test() {
  // CHECK: hello host executor!
  %ch0 = hex.new.chain
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain
  hex.return
}

// CHECK-LABEL: --- Running 'three_print_error'
func @three_print_error() {
  %ch0 = hex.new.chain

  // CHECK: hello host executor!
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain

  // CHECK: hello host executor!
  %ch2 = "tfrt_test.print_hello"(%ch1) : (!hex.chain) -> !hex.chain

  // CHECK: hello host executor!
  %ch3 = "tfrt_test.print_hello"(%ch2) : (!hex.chain) -> !hex.chain

  hex.return
}

// CHECK-LABEL: --- Running 'basic.i1'
func @basic.i1() -> i1 {
  %ch0 = hex.new.chain

  %zero = hex.constant.i1 0
  // CHECK: int1 = 0
  hex.print.i1 %zero, %ch0

  %one = hex.constant.i1 1
  // CHECK: int1 = 1
  hex.print.i1 %one, %ch0

  // CHECK: 'basic.i1' returned 1
  hex.return %one : i1
}

// CHECK-LABEL: --- Running 'basic.i32'
func @basic.i32() -> i32 {
  %x = hex.constant.i32 42

  %ch0 = hex.new.chain

  // CHECK: int32 = 42
  hex.print.i32 %x, %ch0

  %c1 = hex.constant.i32 1
  // CHECK: int32 = 1
  hex.print.i32 %c1, %ch0

  %y = hex.add.i32 %x, %c1
  // CHECK: int32 = 43
  hex.print.i32 %y, %ch0

  %quot, %rem = hex.div.i32 %y, %x
  // CHECK: int32 = 1
  hex.print.i32 %quot, %ch0
  // CHECK: int32 = 1
  hex.print.i32 %rem, %ch0

  %y1, %y2, %y3 = tfrt_test.count3.i32 %y
  // CHECK: int32 = 44
  hex.print.i32 %y1, %ch0

  // CHECK: int32 = 45
  hex.print.i32 %y2, %ch0

  // CHECK: int32 = 46
  hex.print.i32 %y3, %ch0

  %y4 = "tfrt_test.copy.with_delay.i32"(%y3) : (i32) -> i32
  // CHECK: int32 = 46
  hex.print.i32 %y4, %ch0

  // CHECK: 'basic.i32' returned 43
  hex.return %y : i32
}

// CHECK-LABEL: --- Running 'divide_by_zero.i32'
func @divide_by_zero.i32() -> i32 {
  %zero = hex.constant.i32 0
  %x = hex.constant.i32 42

  // expected-error @+1 {{runtime error: Divide by zero}}
  %quot, %rem = hex.div.i32 %x, %zero

  hex.return %quot : i32
}

// CHECK-LABEL: --- Running 'basic.i64'
func @basic.i64() -> i64 {
  %x = hex.constant.i64 42

  %ch0 = hex.new.chain

  // CHECK: int64 = 42
  hex.print.i64 %x, %ch0

  %c1 = hex.constant.i64 1
  // CHECK: int64 = 1
  hex.print.i64 %c1, %ch0

  %y = hex.add.i64 %x, %c1
  // CHECK: int64 = 43
  hex.print.i64 %y, %ch0

  %z = "tfrt_test.copy.with_delay.i64"(%y) : (i64) -> i64
  // CHECK: int64 = 43
  hex.print.i64 %z, %ch0

  // CHECK: 'basic.i64' returned 43
  hex.return %z : i64
}

// CHECK-LABEL: --- Running 'basic.f32'
func @basic.f32() -> f32 {
  %ch0 = hex.new.chain

  %zero = hex.constant.f32 0.0
  // CHECK: f32 = 0.000000
  hex.print.f32 %zero, %ch0

  %one = hex.constant.f32 1.0
  // CHECK: f32 = 1.000000
  hex.print.f32 %one, %ch0

  %six = hex.constant.f32 6.0
  %seven = hex.add.f32 %one, %six
  // CHECK: f32 = 7.000000
  hex.print.f32 %seven, %ch0

  // CHECK: 'basic.f32' returned 1.0
  hex.return %one : f32
}

// CHECK-LABEL: --- Running 'basic.f64'
func @basic.f64() -> f64 {
  %ch0 = hex.new.chain

  %zero = hex.constant.f64 0.0
  // CHECK: f64 = 0.000000
  hex.print.f64 %zero, %ch0

  %one = hex.constant.f64 1.0
  // CHECK: f64 = 1.000000
  hex.print.f64 %one, %ch0

  %six = hex.constant.f64 6.0
  %seven = hex.add.f64 %one, %six
  // CHECK: f64 = 7.000000
  hex.print.f64 %seven, %ch0

  // CHECK: 'basic.f64' returned 1.0
  hex.return %one : f64
}

// CHECK-LABEL: --- Running 'basic_strings'
func @basic_strings() {
  %x = "tfrt_test.get_string"() { value = "bark" } : () -> !hex.string
  %y = "tfrt_test.get_string"() { value = "rarf" } : () -> !hex.string
  %ch0 = hex.new.chain

  // CHECK: string = bark
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!hex.string, !hex.chain) -> (!hex.chain)

  %z = "tfrt_test.append_string"(%x, %y) : (!hex.string, !hex.string) -> !hex.string

  // CHECK: string = barkrarf
  "tfrt_test.print_string"(%z, %ch1) : (!hex.string, !hex.chain) -> (!hex.chain)

  hex.return
}

// CHECK-LABEL: --- Not running 'call_print.i32' because it has arguments
func @call_print.i32(%x: i32) {
  %ch0 = hex.new.chain
  hex.print.i32 %x, %ch0
  hex.return
}

// CHECK-LABEL: --- Not running 'add_one' because it has arguments
func @add_one(%x: i32) -> i32 {
  %c1 = hex.constant.i32 1
  %y = hex.add.i32 %x, %c1
  hex.return %y : i32
}

// CHECK-LABEL: --- Running 'caller'
func @caller() {
  %c1 = hex.constant.i32 1

  // CHECK-NEXT: int32 = 1
  hex.call @call_print.i32(%c1) : (i32) -> ()

  %x = hex.call @add_one(%c1) : (i32) -> i32

  %ch0 = hex.new.chain

  // CHECK-NEXT: int32 = 2
  hex.print.i32 %x, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'test_error_result'
func @test_error_result() -> i32 {
  %x = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}
  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_error_result_concrete_async_success'
func @test_error_result_concrete_async_success() -> i32 {
  %in = hex.constant.i32 0
  %x = "tfrt_test.report_error_concrete_async"(%in) : (i32) -> i32
  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result_concrete_async_success' returned 0

// CHECK-LABEL: --- Running 'test_error_result_concrete_async_fail'
func @test_error_result_concrete_async_fail() -> i32 {
  %in = hex.constant.i32 1
  %x = "tfrt_test.report_error_concrete_async"(%in) : (i32) -> i32 // expected-error {{something bad happened asynchronously}}

  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result_concrete_async_fail' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_error_result_indirect_async_success'
func @test_error_result_indirect_async_success() -> i32 {
  %in = hex.constant.i32 0
  %x = "tfrt_test.report_error_indirect_async"(%in) : (i32) -> i32
  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result_indirect_async_success' returned 0

// CHECK-LABEL: --- Running 'test_error_result_indirect_async_fail'
func @test_error_result_indirect_async_fail() -> i32 {
  %in = hex.constant.i32 1
  %x = "tfrt_test.report_error_indirect_async"(%in) : (i32) -> i32 // expected-error {{something bad happened asynchronously}}

  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result_indirect_async_fail' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_ignore_error'
func @test_ignore_error() -> i32 {
  "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}
  %x = hex.constant.i32 1
  hex.return %x : i32
}
// CHECK: 'test_ignore_error' returned 1

// CHECK-LABEL: --- Running 'test_partial_fail'
func @test_partial_fail() -> !hex.chain {
  // Sets %x to 1 and %y to an error.
  %x, %y = "tfrt_test.partial_fail"() : () -> (i32, i32) // expected-error {{something bad happened}}

  %ch0 = hex.new.chain
  // CHECK: int32 = 1
  hex.print.i32 %x, %ch0
  // This hex.print.i32 does not run. BEFExecutor automatically propagates the
  // error in %y to %ch1.
  %ch1 = hex.print.i32 %y, %ch0

  hex.return %ch1 : !hex.chain
}
// CHECK-NEXT: 'test_partial_fail' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_cancel'
func @test_cancel() -> !hex.chain {
  %ch0 = hex.new.chain
  %x, %ch1 = "tfrt_test.cancel"(%ch0) : (!hex.chain) -> (i32, !hex.chain)

  // The following kernels are skipped due to cancellation.
  // CHECK-NOT: int32 = 0
  %ch2 = hex.print.i32 %x, %ch1

  hex.return %ch2 : !hex.chain
}

// CHECK-NEXT: 'test_cancel' returned <<error: Cancelled>>

// CHECK-LABEL: --- Running 'test_async_value_get'
func @test_async_value_get() -> () {
  %x = "tfrt_test.async_value_get"() : () -> !hex.string

  %ch0 = hex.new.chain
  // CHECK: string = base1:base2:final_class:3
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!hex.string, !hex.chain) -> (!hex.chain)

  hex.return
}

// CHECK-LABEL: --- Running 'test_async_value_ref'
func @test_async_value_ref() -> () {
  %x = "tfrt_test.async_value_ref"() : () -> !hex.string

  %ch0 = hex.new.chain
  // CHECK: string = available:2;unavailable;(3);available:3;unavailable;available:4
  %ch1 = "tfrt_test.print_string"(%x, %ch0) : (!hex.string, !hex.chain) -> (!hex.chain)

  hex.return
}

// CHECK-LABEL: --- Running 'test_logging'
func @test_logging() {
  %x = "tfrt_test.get_string"() { value = "hello world" } : () -> !hex.string
  %c1 = hex.new.chain
  // CHECK: from TFRT_LOG(INFO): hello world
  // CHECK: from TFRT_LOG(WARNING): hello world
  // CHECK: from TFRT_LOG(ERROR): hello world
  // CHECK: from TFRT_LOG_IF(INFO, true): hello world
  // CHECK: from TFRT_LOG_IF(WARNING, true): hello world
  // CHECK: from TFRT_LOG_IF(ERROR, true): hello world
  %c2 = "tfrt_test.logging" (%x, %c1) : (!hex.string, !hex.chain) -> (!hex.chain)
  hex.return
}

// CHECK-LABEL: --- Running 'test_explicit_resource_management'
func @test_explicit_resource_management() -> !hex.chain {
  %c0 = hex.new.chain

  // CHECK-NEXT: Allocated TestResource
  %res, %c1 = "tfrt_test.allocate_resource" (%c0) : (!hex.chain) -> (!hex.test.resource, !hex.chain)

  // test.deallocate_resource explicitly destroys TestResource, so deallocation
  // happens before test.deallocate_resource completes.
  //
  // CHECK-NEXT: Destroyed TestResource
  // CHECK-NEXT: test.deallocate_resource done
  %c2 = "tfrt_test.deallocate_resource" (%res, %c1) : (!hex.test.resource, !hex.chain) -> (!hex.chain)
  hex.return %c2 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_explicit_resource_management_with_cancellation'
func @test_explicit_resource_management_with_cancellation() -> !hex.chain {
  %c0 = hex.new.chain
  // CHECK-NEXT: Allocated TestResource
  %res, %c1 = "tfrt_test.allocate_resource" (%c0) : (!hex.chain) -> (!hex.test.resource, !hex.chain)

  %x, %c2 = "tfrt_test.cancel"(%c1) : (!hex.chain) -> (i32, !hex.chain)

  // Cancellation skips test.deallocate_resource, but TestResource is still
  // destroyed.
  //
  // CHECK-NEXT: Destroyed TestResource
  %c3 = "tfrt_test.deallocate_resource" (%res, %c2) : (!hex.test.resource, !hex.chain) -> (!hex.chain)
  hex.return %c3 : !hex.chain
}
// CHECK-NEXT: 'test_explicit_resource_management_with_cancellation' returned <<error: Cancelled>>

// CHECK-LABEL: --- Running 'test_shared_context'
func @test_shared_context() {
  %ch0 = hex.new.chain

  // use the sample shared context twice to verify only one instance is created.
  %name1, %ch1 = "tfrt_test.use_sample_shared_context"(%ch0) : (!hex.chain) -> (!hex.string, !hex.chain)
  %name2, %ch2 = "tfrt_test.use_sample_shared_context"(%ch0) : (!hex.chain) -> (!hex.string, !hex.chain)

  // CHECK: string = sample_shared_context0
  %ch3 = "tfrt_test.print_string"(%name1, %ch1) : (!hex.string, !hex.chain) -> (!hex.chain)

  // CHECK: string = sample_shared_context0
  %ch4 = "tfrt_test.print_string"(%name2, %ch2) : (!hex.string, !hex.chain) -> (!hex.chain)

  hex.return
}

// CHECK-LABEL: --- Running 'return_multi'
func @return_multi() -> (i32, i32, i32, i32) {
  %a = hex.constant.i32 1
  %b = hex.constant.i32 2

  // CHECK: 'return_multi' returned 1,1,1,2
  hex.return %a, %a, %a, %b : i32, i32, i32, i32
}

func @nested_array() {
  %ch0 = hex.new.chain

  %a, %b, %c, %d = "tfrt_test.flat"() { value = [["string", [1 : i32, 0 : i32]], [1.0 : f32]]} : () -> (!hex.string, i32, i32, f32)

  // CHECK: string = string
  "tfrt_test.print_string"(%a, %ch0) : (!hex.string, !hex.chain) -> (!hex.chain)
  // CHECK: int32 = 1
  hex.print.i32 %b, %ch0
  // CHECK: int32 = 0
  hex.print.i32 %c, %ch0
  // CHECK: f32 = 1.000000
  hex.print.f32 %d, %ch0

  hex.return
}

// CHECK-LABEL: --- Running 'test_error_result_async'
func @test_error_result_async() -> i32 {
  %x = "tfrt_test.report_error_async"() : () -> i32 // expected-error {{something bad happened asynchronously}}
  hex.return %x : i32
}
// CHECK-NEXT: 'test_error_result_async' returned <<error: something bad happened asynchronously>>

// CHECK-LABEL: --- Running 'test_error_result_async_unused'
func @test_error_result_async_unused() -> i32 {
  %x = hex.constant.i32 123
  %y = "tfrt_test.report_error_async"() : () -> i32  // expected-error {{something bad happened asynchronously}}
  // CHECK: 'test_error_result_async_unused' returned 123
  hex.return %x : i32
}
