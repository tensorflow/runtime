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

// RUN: bef_executor_lite --work_queue_type=s %s.bef 2>&1 | FileCheck %s

// The following tests should be executed with single-threaded work queue.

// CHECK-LABEL: --- Not running 'call_add.i32' because it has arguments
func @call_add.i32(%x: i32, %y: i32) -> i32 {
  %z = tfrt.add.i32 %x, %y
  tfrt.return %z : i32
}

// CHECK-LABEL: --- Running 'nonstrict_kernel_with_error_input'
func @nonstrict_kernel_with_error_input() -> i32 {
  %ch0 = tfrt.new.chain

  %one = tfrt.constant.i32 1

  %error = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}

  %async_one = "tfrt_test.async_constant.i32"() { value = 1 : i32 } : () -> i32

  tfrt.print.i32 %async_one, %ch0

  %output = tfrt.call @call_add.i32(%error, %async_one) : (i32, i32) -> i32

  tfrt.return %output : i32
}
// CHECK-NEXT: int32 = 1
// CHECK-NEXT: 'nonstrict_kernel_with_error_input' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'strict_kernel_with_error_input'
func @strict_kernel_with_error_input() -> (i32, !tfrt.chain) {
  %ch0 = tfrt.new.chain

  %one = tfrt.constant.i32 1

  %error = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}

  %async_one = "tfrt_test.async_constant.i32"() { value = 1 : i32 } : () -> i32

  %ch1 = tfrt.print.i32 %async_one, %ch0

  %output = tfrt.add.i32 %error, %async_one

  tfrt.return %output, %ch1 : i32, !tfrt.chain
}
// CHECK-NEXT: int32 = 1
// CHECK-NEXT: 'strict_kernel_with_error_input' returned <<error: something bad happened>>
