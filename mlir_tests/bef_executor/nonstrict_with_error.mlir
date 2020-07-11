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

// CHECK-LABEL: --- Not running 'call_print.i32' because it has arguments
func @call_print.i32 (%x: i32, %y: i32) {
  %ch0 = hex.new.chain
  %ch1 = hex.print.i32 %x, %ch0
  %ch2 = hex.print.i32 %y, %ch0
  hex.return
}

// CHECK-LABEL: --- Running 'call_test'
func @call_test () {
  %v1 = hex.constant.i32 42
  %v2 = hex.constant.i32 64
  %x = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  // CHECK-NEXT: int32 = 42
  hex.call @call_print.i32 (%v1, %x) : (i32, i32) -> ()
  // CHECK-NEXT: int32 = 64
  hex.call @call_print.i32 (%x, %v2) : (i32, i32) -> ()
  hex.return
}

// hex.if is invoked since it's a non-strict kernel, which is executed even when
// one of the inputs is an error. In the test, hex.if's condition is true, so
// %v1 is printed.
// CHECK-LABEL: --- Running 'if_test'
func @if_test () {
  %cond_true = hex.constant.i1 1
  %v1 = hex.constant.i32 384
  %v2 = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  hex.if %cond_true, %v1, %v2 : (i32, i32) -> () {
    %ch0 = hex.new.chain
    // CHECK: int32 = 384
    hex.print.i32 %v1, %ch0
    hex.return
  } else {
    %ch0 = hex.new.chain
    hex.print.i32 %v2, %ch0
    hex.return
  }
  hex.return
}

// This hex.repeat will print 'hello host executor' once. There's no printing in
// the remaining two iterations since the input chains for the two print kernels
// are errors.
// CHECK-LABEL: --- Running 'repeat_test'
func @repeat_test() {
  %ch0 = hex.new.chain
  %v0 = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  %count = hex.constant.i32 3

  hex.repeat.i32 %count, %ch0, %v0 : !hex.chain, i32 {
    %ch2 = hex.print.i32 %v0, %ch0
    // CHECK-NEXT: hello host executor!
    %ch3 = "tfrt_test.print_hello"(%ch0) : (!hex.chain) -> !hex.chain
    hex.return %ch2, %v0 : !hex.chain, i32
  }
  // CHECK-NEXT: int32 = 3
  hex.print.i32 %count, %ch0
  hex.return
}
