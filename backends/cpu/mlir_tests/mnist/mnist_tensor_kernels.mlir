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

// RUN: bef_executor %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'test_tensor_kernels'
func @test_tensor_kernels() {
  %ch0 = tfrt.new.chain

  %zero = "tfrt.constant.i32"() { value = 0 : i32 } : () -> i32
  %one = "tfrt.constant.i32"() { value = 1 : i32 } : () -> i32

  %a = "tfrt_dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 3 : i64] } :
    () -> !t.tensor
  %ch1_0 = "tfrt_dht.set_tensor_with_constant_values.i32"(%a, %ch0)
    { values = [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32] } :
    (!t.tensor, !tfrt.chain) -> !tfrt.chain

  %b = "tfrt_dht.create_uninitialized_tensor.i32.2"() { shape = [3 : i64, 2 : i64] } :
    () -> !t.tensor
  %ch1_1 = "tfrt_dht.set_tensor_with_constant_values.i32"(%b, %ch0)
    { values = [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32] } :
    (!t.tensor, !tfrt.chain) -> !tfrt.chain
  %ch2 = tfrt.merge.chains %ch1_0, %ch1_1 : !tfrt.chain, !tfrt.chain

  %c = "tfrt_dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch3 = "tfrt_test.matmul.i32.2"(%one, %a, %b, %zero, %c, %ch2) :
       (i32, !t.tensor, !t.tensor, i32,
       !t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [1, -1, -1, 1]
  %ch4 = tfrt_dht.print_tensor %c, %ch3

  %d = "tfrt_dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch_relu = "tfrt_test.relu.i32"(%c, %d, %ch4) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [1, 0, 0, 1]
  %ch5 = tfrt_dht.print_tensor %d, %ch_relu

  %e = "tfrt_dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch5_1 = "tfrt_test.add.i32"(%c, %d, %e, %ch5) : (!t.tensor,
    !t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [2, -1, -1, 2]
  %ch6 = tfrt_dht.print_tensor %e, %ch5_1

  %ch7 = "tfrt_test.equal.i32"(%c, %d, %e, %ch6) : (!t.tensor,
    !t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [1, 0, 0, 1]
  %ch8 = tfrt_dht.print_tensor %e, %ch7

  %argmax_c = "tfrt_test.argmax.i32.2"(%c, %ch8) : (!t.tensor, !tfrt.chain) ->
    !t.tensor

  // CHECK: shape = [2], values = [0, 1]
  %ch9 = tfrt_dht.print_tensor %argmax_c, %ch8

  %f = "tfrt_dht.create_uninitialized_tensor.f32.1"() { shape = [2 : i64] } : () ->
     !t.tensor
  %ch10 = "tfrt_dht.set_tensor_with_constant_values.f32"(%f, %ch9)
    { values = [1. : f32, 3. : f32] } :
    (!t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2], values = [1.000000e+00, 3.000000e+00]
  %ch11 = tfrt_dht.print_tensor %f, %ch10

  %reduce_mean_f = "tfrt_test.reduce_mean.f32.1"(%f, %ch11) : (!t.tensor,
    !tfrt.chain) -> !t.tensor

  // CHECK: shape = [], values = [2.000000e+00]
  %c12 = tfrt_dht.print_tensor %reduce_mean_f, %ch11

  tfrt.return
}
