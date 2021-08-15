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

// RUN: bef_executor_lite %s.bef --work_queue_type=mstd 2>&1 | FileCheck %s

module attributes {tfrt.cost_threshold = 10 : i64} {

// CHECK-LABEL: --- Running 'print_thread_id'
func @print_thread_id() -> !tfrt.chain {
  %ch = tfrt.new.chain
  %t = "tfrt_test.get_thread_id"(%ch) : (!tfrt.chain) -> (i32)

  // CHECK: int32 = 0
  %ch0 = tfrt.print.i32 %t, %ch

  %t0 = "tfrt_test.get_thread_id"(%ch0) : (!tfrt.chain) -> (i32)
  %t1 = "tfrt_test.get_thread_id"(%ch0) : (!tfrt.chain) -> (i32)

  // CHECK: int32 = 0
  // CHECK: int32 = 1
  %ch1 = tfrt.print.i32 %t0, %ch0
  %ch2 = tfrt.print.i32 %t1, %ch1

  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'lifo_scheduling'
func @lifo_scheduling() -> !tfrt.chain {
  %c0 = tfrt.constant.i32 1

  // kernel 0 and kernel 2 are always executed consecutively due to lifo scheduling.

  // CHECK: id: 0
  // CHECK-NEXT: id: 2
  %c1 = tfrt_test.test_cost %c0 {id = 0 : i64, _tfrt_cost = 1 : i64} : i32
  %c2 = tfrt_test.test_cost %c0 {id = 1 : i64, _tfrt_cost = 1 : i64} : i32
  %c3 = tfrt_test.test_cost %c1 {id = 2 : i64, _tfrt_cost = 1 : i64} : i32

  %ch = tfrt.merge.chains %c2, %c3 : i32, i32
  tfrt.return %ch: !tfrt.chain
}

}
