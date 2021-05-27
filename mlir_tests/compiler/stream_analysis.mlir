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

// RUN: tfrt_opt -tfrt-print-stream -verify-diagnostics %s

module attributes {tfrt.cost_threshold = 10 : i64} {

// expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
func @stream(%a: i32, %b: i32) -> i32 attributes {tfrt.cost_threshold = 5} {
  // stream 0 cost = 1 (root) + 5 (%a0) + 5 (%a1) + 5 (%a2) + 5 (%result) + 1 (return)
  // stream 1 cost = 5 (%b0) + 5 (%b1) + 5 (%b2)

  // %a0, %a1 and %a2 has data dependencies. So even though each of them
  // are above cost threshold, they are merged to the same stream.
  // expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
  %a0 = tfrt.constant.i32 1
  // expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
  %a1 = "tfrt.add.i32"(%a, %a0) : (i32, i32) -> i32
  // expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
  %a2 = "tfrt.add.i32"(%a, %a1) : (i32, i32) -> i32

  // %b0, %b1 and %b2 has data dependencies. So even though each of them
  // are above cost threshold, they are merged to the same stream.
  // expected-remark@+1 {{stream id: 1, stream cost: 15, parent stream: 0}}
  %b0 = tfrt.constant.i32 2
  // expected-remark@+1 {{stream id: 1, stream cost: 15, parent stream: 0}}
  %b1 = "tfrt.add.i32"(%b, %b0) : (i32, i32) -> i32
  // expected-remark@+1 {{stream id: 1, stream cost: 15, parent stream: 0}}
  %b2 = "tfrt.add.i32"(%b, %b1) : (i32, i32) -> i32

  // %a2 and %b2 are equvalent path from root in costs, so we randomly pick
  // one stream to merge into.
  // expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
  %result = "tfrt.add.i32"(%a2, %b2) : (i32, i32) -> i32

  // expected-remark@+1 {{stream id: 0, stream cost: 22, parent stream: -1}}
  tfrt.return %result : i32
}

// expected-remark@+1 {{stream id: 0, stream cost: 12, parent stream: -1}}
func @no_merge() -> (i32, i32, i32) {
  // %0, %1, and %2 are independent. Since they are above cost threshold,
  // each of them is assigned to a different stream.
  // expected-remark@+1 {{stream id: 0, stream cost: 12, parent stream: -1}}
  %0 = tfrt.constant.i32 0
  // expected-remark@+1 {{stream id: 2, stream cost: 10, parent stream: 0}}
  %1 = tfrt.constant.i32 1
  // expected-remark@+1 {{stream id: 1, stream cost: 10, parent stream: 0}}
  %2 = tfrt.constant.i32 2
  // expected-remark@+1 {{stream id: 0, stream cost: 12, parent stream: -1}}
  tfrt.return %0, %1, %2 : i32, i32, i32
}

// expected-remark@+1 {{stream id: 0, stream cost: 18, parent stream: -1}}
func @merge(%ch0: !tfrt.chain) -> !tfrt.chain {
  // stream 0 cost = 1 (root) + 11 (%ch3) + 4 (%ch5) + 1 (%ch6) + 1 (return)
  // stream 4 cost = 4 (%ch1) + 4 (%ch2) + 4 (%ch4)

  // %ch1, %ch2, %ch3, %ch4, %ch5 are independent operations. Since some
  // of them are below cost threshold, they are merged to form a stream
  // (stream 4) that is barely above the threshold.
  // expected-remark@+1 {{stream id: 4, stream cost: 12, parent stream: 0}}
  %ch1 = tfrt_test.test_cost %ch0 {id = 0 : i64, _tfrt_cost = 4 : i64}
  // expected-remark@+1 {{stream id: 4, stream cost: 12, parent stream: 0}}
  %ch2 = tfrt_test.test_cost %ch0 {id = 1 : i64, _tfrt_cost = 4 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 18, parent stream: -1}}
  %ch3 = tfrt_test.test_cost %ch0 {id = 2 : i64, _tfrt_cost = 11 : i64}
  // expected-remark@+1 {{stream id: 4, stream cost: 12, parent stream: 0}}
  %ch4 = tfrt_test.test_cost %ch0 {id = 3 : i64, _tfrt_cost = 4 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 18, parent stream: -1}}
  %ch5 = tfrt_test.test_cost %ch0 {id = 4 : i64, _tfrt_cost = 4 : i64}

  // Since %ch3 has the highest cost path from root, %ch6 is merged to the
  // stream of %ch3.
  // expected-remark@+1 {{stream id: 0, stream cost: 18, parent stream: -1}}
  %ch6 = tfrt.merge.chains %ch1, %ch2, %ch3, %ch4, %ch5 : !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain
  // expected-remark@+1 {{stream id: 0, stream cost: 18, parent stream: -1}}
  tfrt.return %ch6 : !tfrt.chain
}

// expected-remark@+1 {{stream id: 1, stream cost: 17, parent stream: -1}}
func @merge_limit(%ch0: !tfrt.chain) -> !tfrt.chain attributes {tfrt.upper_cost_threshold = 17} {
  // expected-remark@+1 {{stream id: 1, stream cost: 17, parent stream: -1}}
  %ch1 = tfrt_test.test_cost %ch0 {id = 0 : i64, _tfrt_cost = 8 : i64}
  // expected-remark@+1 {{stream id: 1, stream cost: 17, parent stream: -1}}
  %ch2 = tfrt_test.test_cost %ch1 {id = 1 : i64, _tfrt_cost = 8 : i64}
  // Though %ch2 and %ch3 are dependent ops, they are in different streams because the cost exceed the upper threshold.
  // expected-remark@+1 {{stream id: 0, stream cost: 17, parent stream: 1}}
  %ch3 = tfrt_test.test_cost %ch2 {id = 2 : i64, _tfrt_cost = 8 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 17, parent stream: 1}}
  %ch4 = tfrt_test.test_cost %ch3 {id = 3 : i64, _tfrt_cost = 8 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 17, parent stream: 1}}
  tfrt.return %ch4 : !tfrt.chain
}

// expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
func @merge_inter_dependent_streams(%ch0: !tfrt.chain) -> !tfrt.chain attributes {tfrt.merge_inter_dependent_streams = true} {

  // expected-remark@+1 {{stream id: 3, stream cost: 11, parent stream: 0}}
  %ch1 = tfrt_test.test_cost %ch0 {id = 1 : i64, _tfrt_cost = 5 : i64}
  // expected-remark@+1 {{stream id: 3, stream cost: 11, parent stream: 0}}
  %ch2 = tfrt_test.test_cost %ch0 {id = 2 : i64, _tfrt_cost = 5 : i64}
  // expected-remark@+1 {{stream id: 3, stream cost: 11, parent stream: 0}}
  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain

  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch4 = tfrt_test.test_cost %ch0 {id = 3 : i64, _tfrt_cost = 5 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch5 = tfrt_test.test_cost %ch0 {id = 4 : i64, _tfrt_cost = 5 : i64}
  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch6 = tfrt.merge.chains %ch4, %ch5 : !tfrt.chain, !tfrt.chain
  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch7 = tfrt.merge.chains %ch4, %ch5 : !tfrt.chain, !tfrt.chain
  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch8 = tfrt.merge.chains %ch6, %ch7 : !tfrt.chain, !tfrt.chain

  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  %ch9 = tfrt.merge.chains %ch3, %ch8 : !tfrt.chain, !tfrt.chain
  // expected-remark@+1 {{stream id: 0, stream cost: 16, parent stream: -1}}
  tfrt.return %ch9 : !tfrt.chain
}

}
