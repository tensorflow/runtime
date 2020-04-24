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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=always

// CHECK-LABEL: --- Running 'test_matmul_i32'
func @test_matmul_i32() {
  %ch0 = hex.new.chain

  %path = "tfrt_test.get_string"() {
    value = "backends/common/mlir_tests/compat/eigen/test_data/matmul_i32.btf"
  } : () -> !hex.string

  %zero = hex.constant.i32 0
  %one = hex.constant.i32 1
  %a_index = hex.constant.i32 0
  %b_index = hex.constant.i32 1
  %sol_index = hex.constant.i32 2

  // Shape: [6, 8].
  %a = "btf.read_dense_tensor.i32.2"(%path, %a_index) :
    (!hex.string, i32) -> (!t.tensor)

  // Shape: [8, 7].
  %b = "btf.read_dense_tensor.i32.2"(%path, %b_index) :
    (!hex.string, i32) -> (!t.tensor)

  // Shape: [6, 7].
  %c = dht.create_uninitialized_tensor.i32.2 [6 : i64, 7 : i64]

  // Shape: [6, 7].
  %sol = "btf.read_dense_tensor.i32.2"(%path, %sol_index) :
    (!hex.string, i32) -> (!t.tensor)

  %ch1 = "eigen.matmul.i32"(%one, %a, %b, %zero, %c, %ch0) :
    (i32, !t.tensor, !t.tensor, i32,
    !t.tensor, !hex.chain) -> !hex.chain
  %cmp, %ch2 = dht.tensor_equal.i32 %sol, %c, %ch1

  // CHECK: int1 = 1
  hex.print.i1 %cmp, %ch2

  hex.return
}

// CHECK-LABEL: --- Running 'test_matmul_f32'
func @test_matmul_f32() {
  %ch0 = hex.new.chain

  %path = "tfrt_test.get_string"() {
    value = "backends/common/mlir_tests/compat/eigen/test_data/matmul_f32.btf"
  } : () -> !hex.string

  %zero = hex.constant.f32 0.0
  %one = hex.constant.f32 1.0
  %a_index = hex.constant.i32 0
  %b_index = hex.constant.i32 1
  %sol_index = hex.constant.i32 2

  // Shape: [6, 8].
  %a = "btf.read_dense_tensor.f32.2"(%path, %a_index) :
    (!hex.string, i32) -> (!t.tensor)

  // Shape: [8, 7].
  %b = "btf.read_dense_tensor.f32.2"(%path, %b_index) :
    (!hex.string, i32) -> (!t.tensor)

  // Shape: [6, 7].
  %c = dht.create_uninitialized_tensor.f32.2 [6 : i64, 7 : i64]

  // Shape: [6, 7].
  %sol = "btf.read_dense_tensor.f32.2"(%path, %sol_index) :
    (!hex.string, i32) -> (!t.tensor)

  %ch1 = "eigen.matmul.f32"(%one, %a, %b, %zero, %c, %ch0) :
    (f32, !t.tensor, !t.tensor, f32,
    !t.tensor, !hex.chain) -> !hex.chain

  %cmp, %ch2 = "dht.tensor_allclose.f32"(%sol, %c, %ch1) :
    (!t.tensor, !t.tensor, !hex.chain) -> (i1, !hex.chain)

  // CHECK: int1 = 1
  hex.print.i1 %cmp, %ch2

  hex.return
}
