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


// CHECK-LABEL: --- Running 'basic.i32'
func @basic.i32() -> i32 attributes {tfrt.sync} {
  %x = "tfrt.constant_s.i32"() {value = 42 : i32} : () -> i32
  %y = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32

  %z = "tfrt.add_s.i32"(%x, %y) : (i32, i32) -> i32

  // CHECK: 'basic.i32' returned 43
  tfrt.return %z : i32
}

// CHECK-LABEL: --- Running 'fibonacci.i32'
func @fibonacci.i32() -> i32 attributes {tfrt.sync} {
  %0 = "tfrt.constant_s.i32"() {value = 0 : i32} : () -> i32
  %1 = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32

  %2 = "tfrt.add_s.i32"(%0, %1) : (i32, i32) -> i32
  %3 = "tfrt.add_s.i32"(%1, %2) : (i32, i32) -> i32
  %4 = "tfrt.add_s.i32"(%2, %3) : (i32, i32) -> i32
  %5 = "tfrt.add_s.i32"(%3, %4) : (i32, i32) -> i32
  %6 = "tfrt.add_s.i32"(%4, %5) : (i32, i32) -> i32
  %7 = "tfrt.add_s.i32"(%5, %6) : (i32, i32) -> i32

  // CHECK: 'fibonacci.i32' returned 13
  tfrt.return %7 : i32
}

// CHECK-LABEL: --- Running 'test_fail'
func @test_fail() -> i32 attributes {tfrt.sync} {
  %x = "tfrt_test.fail_s"() : () -> i32 // expected-error {{something bad happened}}
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_fail' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_error_result'
func @test_error_result() -> i32 attributes {tfrt.sync} {
  %x = "tfrt_test.error_s"() : () -> i32 // expected-error {{something bad happened}}
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result' returned <<error: something bad happened>>

// CHECK-LABEL: --- Running 'test_remaining_sync_args'
func @test_remaining_sync_args() -> i32 attributes {tfrt.sync} {
  %0 = "tfrt.constant_s.i32"() {value = 0 : i32} : () -> i32
  %1 = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32
  %2 = "tfrt.constant_s.i32"() {value = 2 : i32} : () -> i32
  %3 = "tfrt.constant_s.i32"() {value = 3 : i32} : () -> i32

  %x = "tfrt_test.sync_sum"(%0, %1, %2, %3) : (i32, i32, i32, i32) -> i32
  // CHECK-NEXT: 'test_remaining_sync_args' returned 6
  tfrt.return %x : i32
}
