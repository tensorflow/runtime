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

// RUN: cat %s.bef | code_size_test_driver | FileCheck %s

// CHECK-LABEL: --- Not running 'fib' because it has arguments
func @fib(%arg: i32) -> i32 {
  %one = tfrt.constant.i32 1
  %cond = "tfrt.lessequal.i32"(%arg, %one) : (i32, i32) -> (i1)
  %res = tfrt.if %cond, %arg : (i32) -> i32 {
    tfrt.return %arg : i32
  } else {
    %min_one = tfrt.constant.i32 -1
    %n_min_one = tfrt.add.i32 %arg, %min_one
    %n_min_two = tfrt.add.i32 %n_min_one, %min_one
    %fib_n_min_one = tfrt.call @fib(%n_min_one) : (i32) -> (i32)
    %fib_n_min_two = tfrt.call @fib(%n_min_two) : (i32) -> (i32)
    %sum = tfrt.add.i32 %fib_n_min_one, %fib_n_min_two
    tfrt.return %sum : i32
  }

  tfrt.return %res : i32
}

// CHECK-LABEL: --- Running 'fib_driver'
func @fib_driver() {
  %zero = tfrt.constant.i32 0
  %fib0 = tfrt.call @fib(%zero) : (i32) -> (i32)

  %four = tfrt.constant.i32 4
  %fib4 = tfrt.call @fib(%four) : (i32) -> (i32)

  %eight = tfrt.constant.i32 8
  %fib8 = tfrt.call @fib(%eight) : (i32) -> (i32)

  %sixteen = tfrt.constant.i32 16
  %fib16 = tfrt.call @fib(%sixteen) : (i32) -> (i32)

  %ch0 = tfrt.new.chain
  // CHECK: int32 = 0
  %ch1= tfrt.print.i32 %fib0, %ch0

  // CHECK: int32 = 3
  %ch2= tfrt.print.i32 %fib4, %ch1

  // CHECK: int32 = 21
  %ch3= tfrt.print.i32 %fib8, %ch2

  // CHECK: int32 = 987
  %ch4= tfrt.print.i32 %fib16, %ch3

  tfrt.return
}
