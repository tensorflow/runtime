// Copyright 2021 The TensorFlow Runtime Authors
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
// RUN: bef_executor_lite -work_queue_type=mstd %s.bef | FileCheck %s --dump-input=fail

func @callee(%ch: !tfrt.chain, %a: i32, %b: i32) -> (!tfrt.chain, i32, i32) attributes {tfrt.cost_threshold = 1 : i64} {
  %ua = "tfrt_test.get_uniqueness"(%a) : (i32) -> (i1)
  %ub = "tfrt_test.get_uniqueness"(%b) : (i32) -> (i1)

  %ch0 = "tfrt_test.print_bool"(%ch, %ua) : (!tfrt.chain, i1) -> (!tfrt.chain)
  %ch1 = "tfrt_test.print_bool"(%ch0, %ub) : (!tfrt.chain, i1) -> (!tfrt.chain)

  tfrt.return %ch1, %a, %b : !tfrt.chain, i32, i32
}

// CHECK-LABEL: test_uniqueness
func @test_uniqueness() -> (!tfrt.chain) {
  %ch = tfrt.new.chain
  %a = tfrt.constant.i32 0
  %b = tfrt.constant.i32 1

  // CHECK: false
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  %ch0, %a0, %b0 = tfrt.call @callee(%ch, %a, %b) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)
  %ch1, %a1, %b1 = tfrt.call @callee(%ch0, %a0, %b0) : (!tfrt.chain, i32, i32) -> (!tfrt.chain, i32, i32)

  %ch2 = tfrt.print.i32 %a1, %ch1
  %ch3 = tfrt.print.i32 %b1, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: test_indirect_uniqueness
func @test_indirect_uniqueness() -> (!tfrt.chain) {
  %ch = tfrt.new.chain
  %a = tfrt.constant.i32 100

  %ia = "tfrt_test.make_indirect"(%a) : (i32) -> (i32)

  // %a has not been used by tfrt.print below yet, so the indirect async value of %a must not be unique at this line.
  %uia = "tfrt_test.get_uniqueness"(%ia) : (i32) -> (i1)

  // CHECK: false
  // CHECK-NEXT: int32 = 100
  %ch0 = "tfrt_test.print_bool"(%ch, %uia) : (!tfrt.chain, i1) -> (!tfrt.chain)
  %ch1 = tfrt.print.i32 %a, %ch0

  tfrt.return %ch1 : !tfrt.chain
}
