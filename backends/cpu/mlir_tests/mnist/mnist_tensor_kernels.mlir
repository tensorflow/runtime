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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'test_tensor_kernels'
func @test_tensor_kernels() {
  %ch0 = hex.new.chain

  %zero = "hex.constant.i32"() { value = 0 : i32 } : () -> i32
  %one = "hex.constant.i32"() { value = 1 : i32 } : () -> i32

  %a = "dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 3 : i64] } :
    () -> !t.tensor
  %ch1_0 = "dht.set_tensor_with_constant_values.i32"(%a, %ch0)
    { values = [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32] } :
    (!t.tensor, !hex.chain) -> !hex.chain

  %b = "dht.create_uninitialized_tensor.i32.2"() { shape = [3 : i64, 2 : i64] } :
    () -> !t.tensor
  %ch1_1 = "dht.set_tensor_with_constant_values.i32"(%b, %ch0)
    { values = [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32] } :
    (!t.tensor, !hex.chain) -> !hex.chain
  %ch2 = hex.merge.chains %ch1_0, %ch1_1

  %c = "dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch3 = "tfrt_test.matmul.i32.2"(%one, %a, %b, %zero, %c, %ch2) :
       (i32, !t.tensor, !t.tensor, i32,
       !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2, 2], values = [1, -1, -1, 1]
  %ch4 = dht.print_tensor %c, %ch3

  %d = "dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch_relu = "tfrt_test.relu.i32"(%c, %d, %ch4) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2, 2], values = [1, 0, 0, 1]
  %ch5 = dht.print_tensor %d, %ch_relu

  %e = "dht.create_uninitialized_tensor.i32.2"() { shape = [2 : i64, 2 : i64] } :
    () -> !t.tensor

  %ch5_1 = "tfrt_test.add.i32"(%c, %d, %e, %ch5) : (!t.tensor,
    !t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2, 2], values = [2, -1, -1, 2]
  %ch6 = dht.print_tensor %e, %ch5_1

  %ch7 = "tfrt_test.equal.i32"(%c, %d, %e, %ch6) : (!t.tensor,
    !t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2, 2], values = [1, 0, 0, 1]
  %ch8 = dht.print_tensor %e, %ch7

  %argmax_c = "tfrt_test.argmax.i32.2"(%c, %ch8) : (!t.tensor, !hex.chain) ->
    !t.tensor

  // CHECK: shape = [2], values = [0, 1]
  %ch9 = dht.print_tensor %argmax_c, %ch8

  %f = "dht.create_uninitialized_tensor.f32.1"() { shape = [2 : i64] } : () ->
     !t.tensor
  %ch10 = "dht.set_tensor_with_constant_values.f32"(%f, %ch9)
    { values = [1. : f32, 3. : f32] } :
    (!t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2], values = [1.000000e+00, 3.000000e+00]
  %ch11 = dht.print_tensor %f, %ch10

  %reduce_mean_f = "tfrt_test.reduce_mean.f32.1"(%f, %ch11) : (!t.tensor,
    !hex.chain) -> !t.tensor

  // CHECK: shape = [], values = [2.000000e+00]
  %c12 = dht.print_tensor %reduce_mean_f, %ch11

  hex.return
}
