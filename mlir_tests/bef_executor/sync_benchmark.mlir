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

func @fibonacci.i32(%0: i32, %1: i32) -> () attributes {tfrt.sync} {
  %2 = "tfrt.add_s.i32"(%0, %1) : (i32, i32) -> i32
  %3 = "tfrt.add_s.i32"(%1, %2) : (i32, i32) -> i32
  %4 = "tfrt.add_s.i32"(%2, %3) : (i32, i32) -> i32
  %5 = "tfrt.add_s.i32"(%3, %4) : (i32, i32) -> i32
  %6 = "tfrt.add_s.i32"(%4, %5) : (i32, i32) -> i32
  %7 = "tfrt.add_s.i32"(%5, %6) : (i32, i32) -> i32
  %8 = "tfrt.add_s.i32"(%6, %7) : (i32, i32) -> i32
  %9 = "tfrt.add_s.i32"(%7, %8) : (i32, i32) -> i32
  %10 = "tfrt.add_s.i32"(%8, %9) : (i32, i32) -> i32
  %11 = "tfrt.add_s.i32"(%9, %10) : (i32, i32) -> i32
  %12 = "tfrt.add_s.i32"(%10, %11) : (i32, i32) -> i32
  %13 = "tfrt.add_s.i32"(%11, %12) : (i32, i32) -> i32
  %14 = "tfrt.add_s.i32"(%12, %13) : (i32, i32) -> i32
  %15 = "tfrt.add_s.i32"(%13, %14) : (i32, i32) -> i32

  tfrt.return
}

// CHECK-LABEL: --- Running 'sync_benchmark'
func @sync_benchmark() attributes {tfrt.sync} {
  // CHECK: BM:fibonacci.i32:Duration(ns):
  // CHECK: BM:fibonacci.i32:Count:
  // CHECK: BM:fibonacci.i32:Time Min(ns):
  // CHECK: BM:fibonacci.i32:Time 50%(ns):
  // CHECK: BM:fibonacci.i32:Time 95%(ns):
  // CHECK: BM:fibonacci.i32:Time 99%(ns):

  %0 = "tfrt.constant_s.i32"() {value = 0 : i32} : () -> i32
  %1 = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32

  tfrt_test.sync_benchmark @fibonacci.i32(%0 : i32, %1 : i32)
      duration_secs = 1, max_count = 1000000, num_warmup_runs = 10

  tfrt.return
}
