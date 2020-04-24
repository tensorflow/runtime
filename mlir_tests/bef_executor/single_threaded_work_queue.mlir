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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor --work_queue_type=s 2>&1 | FileCheck %s --dump-input=fail

// The following tests should be executed with single-threaded work queue.

// CHECK-LABEL: --- Not running 'call_add.i32' because it has arguments
func @call_add.i32(%x: i32, %y: i32) -> i32 {
  %z = hex.add.i32 %x, %y
  hex.return %z : i32
}

// CHECK-LABEL: --- Running 'nonstrict_kernel_with_error_input'
func @nonstrict_kernel_with_error_input() -> i32 {
  %ch0 = hex.new.chain

  %one = hex.constant.i32 1

  %error = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}

  %async_one = "hex.async_constant.i32"() { value = 1 : i32 } : () -> i32

  // This should be printed after hex.return. This is because RunBefExecutor
  // calls Await() before it calls Quiesce(). %output should be available
  // immediately because we make kernel with async error non-strict. %async_one
  // should be available after we call Quiesce().
  hex.print.i32 %async_one, %ch0

  // hex.add should be ready for processing before %async_one is available.
  %output = hex.call @call_add.i32(%error, %async_one) : (i32, i32) -> i32

  hex.return %output : i32
}
// CHECK-NEXT: 'nonstrict_kernel_with_error_input' returned <<error: something bad happened>>
// CHECK-NEXT: int32 = 1

// CHECK-LABEL: --- Running 'strict_kernel_with_error_input'
func @strict_kernel_with_error_input() -> i32 {
  %ch0 = hex.new.chain

  %one = hex.constant.i32 1

  %error = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}

  %async_one = "hex.async_constant.i32"() { value = 1 : i32 } : () -> i32

  // This should be printed after hex.return. This is because RunBefExecutor
  // calls Await() before it calls Quiesce(). %output should be available
  // immediately because we make kernel with async error non-strict. %async_one
  // should be available after we call Quiesce().
  hex.print.i32 %async_one, %ch0

  // hex.add should be ready for processing before %async_one is available.
  %output = hex.add.i32 %error, %async_one

  hex.return %output : i32
}
// CHECK-NEXT: 'strict_kernel_with_error_input' returned <<error: something bad happened>>
// CHECK-NEXT: int32 = 1
