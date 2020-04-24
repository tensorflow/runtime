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

// RUN: tfrt_translate -mlir-to-bef %s | code_size_test_driver | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Not running 'fib' because it has arguments
func @fib(%arg: i32) -> i32 {
  %one = hex.constant.i32 1
  %cond = "hex.lessequal.i32"(%arg, %one) : (i32, i32) -> (i1)
  %res = hex.if %cond, %arg : (i32) -> i32 {
    hex.return %arg : i32
  } else {
    %min_one = hex.constant.i32 -1
    %n_min_one = hex.add.i32 %arg, %min_one
    %n_min_two = hex.add.i32 %n_min_one, %min_one
    %fib_n_min_one = hex.call @fib(%n_min_one) : (i32) -> (i32)
    %fib_n_min_two = hex.call @fib(%n_min_two) : (i32) -> (i32)
    %sum = hex.add.i32 %fib_n_min_one, %fib_n_min_two
    hex.return %sum : i32
  }

  hex.return %res : i32
}

// CHECK-LABEL: --- Running 'fib_driver'
func @fib_driver() {
  %zero = hex.constant.i32 0
  %fib0 = hex.call @fib(%zero) : (i32) -> (i32)

  %four = hex.constant.i32 4
  %fib4 = hex.call @fib(%four) : (i32) -> (i32)

  %eight = hex.constant.i32 8
  %fib8 = hex.call @fib(%eight) : (i32) -> (i32)

  %sixteen = hex.constant.i32 16
  %fib16 = hex.call @fib(%sixteen) : (i32) -> (i32)

  %ch0 = hex.new.chain
  // CHECK: int32 = 0
  %ch1= hex.print.i32 %fib0, %ch0

  // CHECK: int32 = 3
  %ch2= hex.print.i32 %fib4, %ch1

  // CHECK: int32 = 21
  %ch3= hex.print.i32 %fib8, %ch2

  // CHECK: int32 = 987
  %ch4= hex.print.i32 %fib16, %ch3

  hex.return
}
