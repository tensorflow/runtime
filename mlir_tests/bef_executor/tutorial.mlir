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

// RUN: bef_executor_lite %s.bef 2>&1 | FileCheck %s

// CHECK: --- Running 'hello'
func @hello() {
  %chain = tfrt.new.chain

  // Create a string containing "hello world" and store it in %hello.
  %hello = "tfrt_test.get_string"() { value = "hello world" } : () -> !tfrt.string

  // Print the string in %hello.
  // CHECK: string = hello world
  "tfrt_test.print_string"(%hello, %chain) : (!tfrt.string, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK: --- Running 'hello_integers'
func @hello_integers() {
  %chain = tfrt.new.chain

  // Create an integer containing 42.
  %forty_two = tfrt.constant.i32 42

  // Print 42.
  // CHECK: int32 = 42
  tfrt.print.i32 %forty_two, %chain

  tfrt.return
}

// CHECK: --- Running 'print_coordinate'
func @print_coordinate() {
  %chain = tfrt.new.chain

  %two = tfrt.constant.i32 2
  %four = tfrt.constant.i32 4

  %coordinate = "tfrt_tutorial.create_coordinate"(%two, %four) : (i32, i32) -> !my.coordinate

  // CHECK: (2, 4)
  "tfrt_tutorial.print_coordinate"(%coordinate, %chain) : (!my.coordinate, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
