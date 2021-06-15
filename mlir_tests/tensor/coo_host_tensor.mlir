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

// CHECK-LABEL: --- Running 'basic_tensor'
func @basic_tensor() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %a, %c0 4 : i32
  %s1, %c2 = coo.convert_dht_to_coo.i32.2 %a, %c1

  // CHECK: dtype = i32, shape = [3, 2], indices = [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2], values = [4, 4, 4, 4, 4, 4]
  %c3 = tfrt_dht.print_tensor %s1, %c2

  %z = tfrt_dht.create_uninitialized_tensor.i32.2 [2 : i64, 3 : i64]
  %c4 = tfrt_dht.fill_tensor_with_constant.i32 %z, %c3 0 : i32
  %s2, %c5 = coo.convert_dht_to_coo.i32.2 %z, %c4

  // CHECK: CooHostTensor dtype = i32, shape = [2, 3], indices = [], values = []
  %c6 = tfrt_dht.print_tensor %s2, %c5

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_roundtrip'
func @tensor_roundtrip() {
  %c1 = tfrt.new.chain

  // Keep tensor uninitialized. This means we'll be getting random values in
  // there.
  %a = tfrt_dht.create_uninitialized_tensor.i32.2 [10 : i64, 10 : i64]
  %s, %c2 = coo.convert_dht_to_coo.i32.2 %a, %c1
  %d, %c3 = coo.convert_coo_to_dht.i32.2 %s, %c2
  %f, %c4 = coo.convert_dht_to_coo.i32.2 %d, %c3

  %cmp, %c5 = tfrt_dht.tensor_equal.i32 %a, %d, %c3
  %cmp2, %c6 = coo.tensor_equal.i32.2 %s, %f, %c4

  // CHECK: int1 = 1
  "tfrt.print.i1"(%cmp, %c5) : (i1, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: int1 = 1
  "tfrt.print.i1"(%cmp2, %c6) : (i1, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// Testing tensor_equal.
// CHECK-LABEL: --- Running 'tensor_equal'
func @tensor_equal() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %a, %c0 4 : i32
  %s1, %c2 = coo.convert_dht_to_coo.i32.2 %a, %c1

  %cmp, %c3 = coo.tensor_equal.i32.2 %s1, %s1, %c2

  // CHECK: int1 = 1
  "tfrt.print.i1"(%cmp, %c3) : (i1, !tfrt.chain) -> (!tfrt.chain)

  %z = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c4 = tfrt_dht.fill_tensor_with_constant.i32 %z, %c0 1 : i32
  %s2, %c5 = coo.convert_dht_to_coo.i32.2 %z, %c4

  %cmp2, %c6 = coo.tensor_equal.i32.2 %s1, %s2, %c5

  // CHECK: int1 = 0
  "tfrt.print.i1"(%cmp2, %c6) : (i1, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return
}

// CHECK-LABEL: --- Running 'basic_f32_tensor'
func @basic_f32_tensor() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %c0 1.0 : f32
  %s1, %c2 = coo.convert_dht_to_coo.f32.2 %a, %c1

  // CHECK: CooHostTensor dtype = f32, shape = [2, 2], indices = [0, 0, 0, 1, 1, 0, 1, 1], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c3 = tfrt_dht.print_tensor %s1, %c2

  tfrt.return
}
