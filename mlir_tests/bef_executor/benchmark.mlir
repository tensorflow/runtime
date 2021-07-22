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

// RUN: bef_executor_lite %s.bef | FileCheck %s --dump-input=fail
// RUN: tfrt_opt %s | tfrt_opt

// A function to demonstrate the use of benchmark kernels.

// CHECK-LABEL: --- Running 'benchmark'
func @benchmark() {
  // CHECK: BM:add.i32:Duration(ns):
  // CHECK: BM:add.i32:Count: 100
  // CHECK: BM:add.i32:Time Min(ns):
  // CHECK: BM:add.i32:Time 50%(ns):
  // CHECK: BM:add.i32:Time 95%(ns):
  // CHECK: BM:add.i32:Time 99%(ns):
  // CHECK: BM:add.i32:CPU Min(ns):
  // CHECK: BM:add.i32:CPU 50%(ns):
  // CHECK: BM:add.i32:CPU 95%(ns):
  // CHECK: BM:add.i32:CPU 99%(ns):
  // CHECK: BM:add.i32:CPU utilization(percent):


  tfrt_test.benchmark "add.i32"() duration_secs = 1, max_count = 100, num_warmup_runs = 10
  {
    %c = tfrt.constant.i32 42
    %x = tfrt.add.i32 %c, %c
    tfrt.return %x : i32
  }

  tfrt.return
}

// A function to demonstrate the use of benchmark kernels with the input compute
// as an external arguments.
func @benchmark2() {
  // CHECK: BM:add.i32:Duration(ns):
  // CHECK: BM:add.i32:Count: 100
  // CHECK: BM:add.i32:Time Min(ns):
  // CHECK: BM:add.i32:Time 50%(ns):
  // CHECK: BM:add.i32:Time 95%(ns):
  // CHECK: BM:add.i32:Time 99%(ns):
  // CHECK: BM:add.i32:CPU Min(ns):
  // CHECK: BM:add.i32:CPU 50%(ns):
  // CHECK: BM:add.i32:CPU 95%(ns):
  // CHECK: BM:add.i32:CPU 99%(ns):
  // CHECK: BM:add.i32:CPU utilization(percent)

  %c = tfrt.constant.i32 42

  // Pass the argument to the function to be benchmarked.
  tfrt_test.benchmark "add.i32"(%c : i32) duration_secs = 1, max_count = 100
  {
    %x = tfrt.add.i32 %c, %c
    tfrt.return %x : i32
  }

  tfrt.return
}
