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
// RUN: tfrt_opt %s | tfrt_opt

// CHECK-LABEL: --- Running 'basic_tensor'
func @basic_tensor() {
  %c0 = hex.new.chain

  %a = dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = dht.fill_tensor_with_constant.i32 %a, %c0 0 : i32

  // CHECK: shape = [3, 2], values = [0, 0, 0, 0, 0, 0]
  %c2 = dht.print_tensor %a, %c1

  %b = dht.create_uninitialized_tensor.i32.0 []
  %c3 = dht.fill_tensor_with_constant.i32 %b, %c2 0 : i32

  // CHECK: shape = [], values = [0]
  %c4 = dht.print_tensor %b, %c3

  %c5 = dht.fill_tensor_with_constant.i32 %a, %c4 1 : i32

  // CHECK: shape = [3, 2], values = [1, 1, 1, 1, 1, 1]
  %c6 = dht.print_tensor %a, %c5

  %c7 = dht.set_tensor_with_constant_values.i32 %a, %c6
    [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32]

  // CHECK: shape = [3, 2], values = [1, -1, 1, -1, 1, -1]
  %c8 = dht.print_tensor %a, %c7

  %buf, %c9 = dht.get_buffer %a, %c8

  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=24>
  %c10 = dht.print_buffer %buf, %c9

  hex.return
}

// Testing tensor_equal.
// CHECK-LABEL: --- Running 'tensor_equal'
func @tensor_equal() {
  %c0 = hex.new.chain

  %a = dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = dht.fill_tensor_with_constant.i32 %a, %c0 0 : i32

  %cmp, %c2 = dht.tensor_equal.i32 %a, %a, %c1

  // CHECK: int1 = 1
  hex.print.i1 %cmp, %c0

  %b = dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c3 = dht.fill_tensor_with_constant.i32 %b, %c0 1 : i32

  %cmp2, %c4 = dht.tensor_equal.i32 %a, %b, %c3

  // CHECK: int1 = 0
  hex.print.i1 %cmp2, %c4

  hex.return
}

// CHECK-LABEL: --- Running 'basic_f32_tensor'
func @basic_f32_tensor() {
  %c0 = hex.new.chain

  %a = dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %c1 = dht.fill_tensor_with_constant.f32 %a, %c0 1.0 : f32

  // CHECK: shape = [2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c2 = dht.print_tensor %a, %c1

  hex.return
}

// CHECK-LABEL: --- Running 'tensor_from_buffer'
func @tensor_from_buffer() {
  %c0 = hex.new.chain

  %a = dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %c1 = dht.fill_tensor_with_constant.f32 %a, %c0 1.0 : f32

  // CHECK: shape = [2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c2 = dht.print_tensor %a, %c1

  %buf, %c3 = dht.get_buffer %a, %c1
  %shape = ts.build_shape [4 : i64, 1 : i64]
  %b, %c4 = dht.make_tensor.f32 %buf, %shape, %c1

  // CHECK: shape = [4, 1], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c5 = dht.print_tensor %b, %c1

  hex.return
}

// CHECK-LABEL: --- Running 'slice_tensor'
func @slice_tensor() {
  %ch0 = hex.new.chain
  %zero = hex.constant.i64 0
  %one = hex.constant.i64 1

  %slice_begin = "dht.create_uninitialized_tensor.i64.1"() { shape = [3 : i64] } : () -> !t.tensor
  %ch1 = "dht.set_tensor_with_values.i64"(%slice_begin, %ch0, %zero, %one, %one)
    : (!t.tensor, !hex.chain, i64, i64, i64) -> !hex.chain

  %input = "dht.create_uninitialized_tensor.i32.3"() { shape = [2 : i64, 3 : i64, 2 : i64] } : () -> !t.tensor
  %ch2 = "dht.set_tensor_with_constant_values.i32"(%input, %ch1)
    { values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32] }
    : (!t.tensor, !hex.chain) -> !hex.chain

  %output = "dht.create_uninitialized_tensor.i32.3"() { shape = [2 : i64, 2 : i64, 1 : i64] } : () -> !t.tensor
  %ch3 = "dht.slice_inplace.i32.3"(%input, %slice_begin, %ch2, %output)
    : (!t.tensor, !t.tensor, !hex.chain, !t.tensor) -> !hex.chain

  // CHECK: shape = [2, 2, 1], values = [4, 6, 10, 12]
  dht.print_tensor %output, %ch3
  hex.return
}

// CHECK-LABEL: --- Running 'bool_tensor'
func @bool_tensor() {
  %ch0 = hex.new.chain

  %value = "dht.create_uninitialized_tensor.bool.1"() { shape = [2 : i64] } : () -> !t.tensor
  %ch1 = "dht.set_tensor_with_constant_values.bool"(%value, %ch0)
    { values = [false, true] } : (!t.tensor, !hex.chain) -> !hex.chain

  // CHECK: shape = [2], values = [0, 1]
  %ch2 = dht.print_tensor %value, %ch1

  hex.return
}

// CHECK-LABEL: --- Running 'dense_attr'
func @dense_attr() {
  %c0 = hex.new.chain

  %a = "tfrt_test.const_dense_attr"() {value = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>} : () -> !t.tensor

  // CHECK: shape = [2, 2], values = [1, 1, 2, 2]
  %c1 = dht.print_tensor %a, %c0

  hex.return
}
