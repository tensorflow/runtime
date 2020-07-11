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
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.print.i32 %x, %ch0
  %ch2 = tfrt.print.i32 %y, %ch0
  tfrt.return
}

// CHECK-LABEL: --- Running 'call_test'
func @call_test () {
  %v1 = tfrt.constant.i32 42
  %v2 = tfrt.constant.i32 64
  %x = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  // CHECK-NEXT: int32 = 42
  tfrt.call @call_print.i32 (%v1, %x) : (i32, i32) -> ()
  // CHECK-NEXT: int32 = 64
  tfrt.call @call_print.i32 (%x, %v2) : (i32, i32) -> ()
  tfrt.return
}

// tfrt.if is invoked since it's a non-strict kernel, which is executed even when
// one of the inputs is an error. In the test, tfrt.if's condition is true, so
// %v1 is printed.
// CHECK-LABEL: --- Running 'if_test'
func @if_test () {
  %cond_true = tfrt.constant.i1 1
  %v1 = tfrt.constant.i32 384
  %v2 = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  tfrt.if %cond_true, %v1, %v2 : (i32, i32) -> () {
    %ch0 = tfrt.new.chain
    // CHECK: int32 = 384
    tfrt.print.i32 %v1, %ch0
    tfrt.return
  } else {
    %ch0 = tfrt.new.chain
    tfrt.print.i32 %v2, %ch0
    tfrt.return
  }
  tfrt.return
}

// This tfrt.repeat will print 'hello host executor' once. There's no printing in
// the remaining two iterations since the input chains for the two print kernels
// are errors.
// CHECK-LABEL: --- Running 'repeat_test'
func @repeat_test() {
  %ch0 = tfrt.new.chain
  %v0 = "tfrt_test.fail" () : () -> i32 // expected-error {{something bad happened}}
  %count = tfrt.constant.i32 3

  tfrt.repeat.i32 %count, %ch0, %v0 : !tfrt.chain, i32 {
    %ch2 = tfrt.print.i32 %v0, %ch0
    // CHECK-NEXT: hello host executor!
    %ch3 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain
    tfrt.return %ch2, %v0 : !tfrt.chain, i32
  }
  // CHECK-NEXT: int32 = 3
  tfrt.print.i32 %count, %ch0
  tfrt.return
}
