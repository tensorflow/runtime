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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: tfrt_opt %s | tfrt_opt

// A function to demonstrate the use of benchmark kernels.

// CHECK-LABEL: --- Running 'benchmark'
func @benchmark() {
  // CHECK: BM:add.i32:Duration(us):
  // CHECK: BM:add.i32:Count:
  // CHECK: BM:add.i32:Time Min(us):
  // CHECK: BM:add.i32:Time 50%(us):
  // CHECK: BM:add.i32:Time 95%(us):
  // CHECK: BM:add.i32:Time 99%(us):

  tfrt_test.benchmark "add.i32"() duration_secs = 1, max_count = 100, num_warmup_runs = 10
  {
    %c = hex.constant.i32 42
    %x = hex.add.i32 %c, %c
    hex.return %x : i32
  }

  hex.return
}

// A function to demonstrate the use of benchmark kernels with the input compute
// as an external arguments.
func @benchmark2() {
  // CHECK: BM:add.i32:Duration(us):
  // CHECK: BM:add.i32:Count:
  // CHECK: BM:add.i32:Time Min(us):
  // CHECK: BM:add.i32:Time 50%(us):
  // CHECK: BM:add.i32:Time 95%(us):
  // CHECK: BM:add.i32:Time 99%(us):

  %c = hex.constant.i32 42

  // Pass the argument to the function to be benchmarked.
  tfrt_test.benchmark "add.i32"(%c : i32) duration_secs = 1, max_count = 100
  {
    %x = hex.add.i32 %c, %c
    hex.return %x : i32
  }

  hex.return
}
