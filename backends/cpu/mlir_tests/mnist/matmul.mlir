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

// TODO: Initialize test matrices inline when b/140124913 is complete.
// TODO(doak): Generate benchmark matrices with a random kernel when it is
//             available, or with a fixture generator, similar to
//             btf/generate_fixtures.py.

// CHECK-LABEL: --- Running 'test_i32'
func @test_i32() {
  %ch0 = tfrt.new.chain
  %path = "tfrt_test.get_string"() {
    value = "backends/cpu/mlir_tests/mnist/test_data/matmul_test_i32.btf"
  } : () -> !tfrt.string
  %zero = tfrt.constant.i32 0
  %one = tfrt.constant.i32 1
  %a_index = tfrt.constant.i32 0
  %b_index = tfrt.constant.i32 1
  %sol_index = tfrt.constant.i32 2

  // Shape: [6, 8].
  %a = "btf.read_dense_tensor.i32.2"(%path, %a_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [8, 7].
  %b = "btf.read_dense_tensor.i32.2"(%path, %b_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  %c = tfrt_dht.create_uninitialized_tensor.i32.2 [6 : i64, 7 : i64]
  // Shape: [6, 7].
  %sol = "btf.read_dense_tensor.i32.2"(%path, %sol_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  %ch1 = "tfrt_test.matmul.i32.2"(%one, %a, %b, %zero, %c, %ch0) :
    (i32, !t.tensor, !t.tensor, i32,
    !t.tensor, !tfrt.chain) -> !tfrt.chain
  %cmp, %ch2 = tfrt_dht.tensor_equal.i32 %sol, %c, %ch1

  // CHECK: int1 = 1
  tfrt.print.i1 %cmp, %ch2

  tfrt.return
}

func @test_f32() {
  %ch0 = tfrt.new.chain
  %path = "tfrt_test.get_string"() {
    value = "backends/cpu/mlir_tests/mnist/test_data/matmul_test_f32.btf"
  } : () -> !tfrt.string
  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0
  %a_index = tfrt.constant.i32 0
  %b_index = tfrt.constant.i32 1
  %sol_index = tfrt.constant.i32 2

  // Shape: [6, 8].
  %a = "btf.read_dense_tensor.f32.2"(%path, %a_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [8, 7].
  %b = "btf.read_dense_tensor.f32.2"(%path, %b_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  %c = tfrt_dht.create_uninitialized_tensor.f32.2 [6 : i64, 7 : i64]
  // Shape: [6, 7].
  %sol = "btf.read_dense_tensor.f32.2"(%path, %sol_index) :
    (!tfrt.string, i32) -> (!t.tensor)
  %ch1 = "tfrt_test.matmul.f32.2"(%one, %a, %b, %zero, %c, %ch0) :
    (f32, !t.tensor, !t.tensor, f32,
    !t.tensor, !tfrt.chain) -> !tfrt.chain
  %cmp, %ch2 = "tfrt_dht.tensor_allclose.f32"(%sol, %c, %ch1) :
    (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  tfrt.print.i1 %cmp, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_test_matmul_f32'
func @BM_test_matmul_f32() {
  %ch0 = tfrt.new.chain
  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0
  // Shape: [512, 512].
  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %ch0 1.0 : f32
  %b = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %b, %ch0 1.0 : f32
  %c = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain

  tfrt_test.benchmark "BM_test_matmul_f32"(
      %zero : f32,
      %one : f32,
      %a : !t.tensor,
      %b : !t.tensor,
      %c : !t.tensor,
      %ch3 : !tfrt.chain)
     duration_secs = 5,
     max_count = 1000,
     num_warmup_runs = 10 {
      %ch_out = "tfrt_test.matmul.f32.2"(%one, %a, %b, %zero, %c, %ch3) :
        (f32, !t.tensor, !t.tensor, f32,
        !t.tensor, !tfrt.chain) -> !tfrt.chain
      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}
