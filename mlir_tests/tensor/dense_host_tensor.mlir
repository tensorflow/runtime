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
// RUN: tfrt_opt %s | tfrt_opt

// CHECK-LABEL: --- Running 'basic_tensor'
func @basic_tensor() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %a, %c0 0 : i32

  // CHECK: shape = [3, 2], values = [0, 0, 0, 0, 0, 0]
  %c2 = tfrt_dht.print_tensor %a, %c1

  %b = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %c3 = tfrt_dht.fill_tensor_with_constant.i32 %b, %c2 0 : i32

  // CHECK: shape = [], values = [0]
  %c4 = tfrt_dht.print_tensor %b, %c3

  %c5 = tfrt_dht.fill_tensor_with_constant.i32 %a, %c4 1 : i32

  // CHECK: shape = [3, 2], values = [1, 1, 1, 1, 1, 1]
  %c6 = tfrt_dht.print_tensor %a, %c5

  %c7 = tfrt_dht.set_tensor_with_constant_values.i32 %a, %c6
    [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32]

  // CHECK: shape = [3, 2], values = [1, -1, 1, -1, 1, -1]
  %c8 = tfrt_dht.print_tensor %a, %c7

  %buf, %c9 = tfrt_dht.get_buffer %a, %c8

  // CHECK: HostBuffer<pointer={{0x[[:xdigit:]]*}}, size=24>
  %c10 = tfrt_dht.print_buffer %buf, %c9

  tfrt.return
}

// Testing tensor_equal.
// CHECK-LABEL: --- Running 'tensor_equal'
func @tensor_equal() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %a, %c0 0 : i32

  %cmp, %c2 = tfrt_dht.tensor_equal.i32 %a, %a, %c1

  // CHECK: int1 = 1
  %ch = tfrt.print.i1 %cmp, %c0

  %b = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %c3 = tfrt_dht.fill_tensor_with_constant.i32 %b, %c0 1 : i32

  %cmp2, %c4 = tfrt_dht.tensor_equal.i32 %a, %b, %c3

  // CHECK: int1 = 0
  tfrt.print.i1 %cmp2, %ch

  tfrt.return
}

// CHECK-LABEL: --- Running 'basic_f32_tensor'
func @basic_f32_tensor() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %c0 1.0 : f32

  // CHECK: shape = [2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c2 = tfrt_dht.print_tensor %a, %c1

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_from_buffer'
func @tensor_from_buffer() {
  %c0 = tfrt.new.chain

  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [2 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %c0 1.0 : f32

  // CHECK: shape = [2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c2 = tfrt_dht.print_tensor %a, %c1

  %buf, %c3 = tfrt_dht.get_buffer %a, %c1
  %shape = ts.build_shape [4 : i64, 1 : i64]
  %b, %c4 = tfrt_dht.make_tensor.f32 %buf, %shape, %c1

  // CHECK: shape = [4, 1], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %c5 = tfrt_dht.print_tensor %b, %c2

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_from_slices'
func @tensor_from_slices() {
  %c0 = tfrt.new.chain

  %buf_size = tfrt.constant.i64 16
  %buf_alignment = tfrt.constant.i64 4
  %buf = tfrt_dht.allocate_buffer %buf_size, %buf_alignment

  %shape = ts.build_shape [2 : i64, 1 : i64]
  %size = tfrt.constant.i64 8

  %buf_a_offset = tfrt.constant.i64 0
  %buf_a = tfrt_dht.get_buffer_slice %buf, %buf_a_offset, %size
  %a, %c1 = tfrt_dht.make_tensor.f32 %buf_a, %shape, %c0
  %c2 = tfrt_dht.fill_tensor_with_constant.f32 %a, %c1 1.0 : f32

  // CHECK: shape = [2, 1], values = [1.000000e+00, 1.000000e+00]
  %c3 = tfrt_dht.print_tensor %a, %c2

  %buf_b_offset = tfrt.constant.i64 8
  %buf_b = tfrt_dht.get_buffer_slice %buf, %buf_b_offset, %size
  %b, %c4 = tfrt_dht.make_tensor.f32 %buf_b, %shape, %c3
  %c5 = tfrt_dht.fill_tensor_with_constant.f32 %b, %c4 2.0 : f32

  // CHECK: shape = [2, 1], values = [2.000000e+00, 2.000000e+00]
  %c6 = tfrt_dht.print_tensor %b, %c5

  %buf_c_shape = ts.build_shape [1 : i64, 1 : i64]
  %buf_c_size = tfrt.constant.i64 4
  %buf_c_offset = tfrt.constant.i64 4
  %buf_c = tfrt_dht.get_buffer_slice %buf, %buf_c_offset, %buf_c_size
  %c, %c7 = tfrt_dht.make_tensor.f32 %buf_c, %buf_c_shape, %c6
  %c8 = tfrt_dht.fill_tensor_with_constant.f32 %c, %c7 3.0 : f32

  // CHECK: shape = [2, 1], values = [1.000000e+00, 3.000000e+00]
  %c9 = tfrt_dht.print_tensor %a, %c8
  // CHECK: shape = [2, 1], values = [2.000000e+00, 2.000000e+00]
  %c10 = tfrt_dht.print_tensor %b, %c9
  // CHECK: shape = [1, 1], values = [3.000000e+00]
  %c11 = tfrt_dht.print_tensor %c, %c10

  tfrt.return
}

// CHECK-LABEL: --- Running 'slice_tensor'
func @slice_tensor() {
  %ch0 = tfrt.new.chain
  %zero = tfrt.constant.i64 0
  %one = tfrt.constant.i64 1

  %slice_begin = tfrt_dht.create_uninitialized_tensor.i64.1 [3 : i64]
  %ch1 = "tfrt_dht.set_tensor_with_values.i64"(%slice_begin, %ch0, %zero, %one, %one)
    : (!t.tensor, !tfrt.chain, i64, i64, i64) -> !tfrt.chain

  %input = tfrt_dht.create_uninitialized_tensor.i32.3 [2 : i64, 3 : i64, 2 : i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.i32 %input, %ch1 [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32]

  %output = tfrt_dht.create_uninitialized_tensor.i32.3 [2 : i64, 2 : i64, 1 : i64]
  %ch3 = "tfrt_test.slice_inplace.i32.3"(%input, %slice_begin, %ch2, %output)
    : (!t.tensor, !t.tensor, !tfrt.chain, !t.tensor) -> !tfrt.chain

  // CHECK: shape = [2, 2, 1], values = [4, 6, 10, 12]
  tfrt_dht.print_tensor %output, %ch3
  tfrt.return
}

// CHECK-LABEL: --- Running 'bool_tensor'
func @bool_tensor() {
  %ch0 = tfrt.new.chain

  %value = tfrt_dht.create_uninitialized_tensor.bool.1 [2 : i64]
  %ch1 = tfrt_dht.set_tensor_with_constant_values.bool %value, %ch0 [false, true]

  // CHECK: shape = [2], values = [0, 1]
  %ch2 = tfrt_dht.print_tensor %value, %ch1

  tfrt.return
}

// CHECK-LABEL: --- Running 'dense_attr'
func @dense_attr() {
  %c0 = tfrt.new.chain

  %a = "tfrt_test.const_dense_attr"() {value = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>} : () -> !t.tensor

  // CHECK: shape = [2, 2], values = [1, 1, 2, 2]
  %c1 = tfrt_dht.print_tensor %a, %c0

  tfrt.return
}

// CHECK-LABEL: --- Running 'sync_basic_tensor'
func @sync_basic_tensor() attributes {tfrt.sync} {
  %a = tfrt_dht_sync.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  tfrt_dht_sync.set_tensor_with_constant_values.i32 %a
    [1 : i32, -1 : i32, 1 : i32, -1 : i32, 1 : i32, -1 : i32]

  // CHECK: shape = [3, 2], values = [1, -1, 1, -1, 1, -1]
  tfrt_dht_sync.print_tensor %a

  %b = "tfrt_test.sync.const_dense_attr"() {value = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>} : () -> !t.tensor

  // CHECK: shape = [2, 2], values = [1, 1, 2, 2]
  tfrt_dht_sync.print_tensor %b

  tfrt.return
}

// CHECK-LABEL: --- Running 'get_tensor_shape'
func @get_tensor_shape() {
  %ch = tfrt.new.chain

  %a = "tfrt_test.const_dense_attr"() {value = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>} : () -> !t.tensor

  %b = tfrt_dht.get_tensor_shape %a
  // CHECK: shape = [2, 2]
  ts.print_shape %b, %ch

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_type_parsing'
func @test_type_parsing() {
  %c0 = tfrt.new.chain

  %buf_size = tfrt.constant.i64 16
  %buf_alignment = tfrt.constant.i64 4
  %buf = "tfrt_dht.allocate_buffer"(%buf_size, %buf_alignment) : (i64, i64) -> !ht.host_buffer

  %shape = ts.build_shape [2 : i64, 1 : i64]
  %size = tfrt.constant.i64 8

  %buf_a_offset = tfrt.constant.i64 0
  %buf_a = tfrt_dht.get_buffer_slice %buf, %buf_a_offset, %size
  %a, %c1 = "tfrt_dht.make_tensor.f32"(%buf_a, %shape, %c0) : (!ht.host_buffer, !ts.shape, !tfrt.chain) -> (!t.tensor, !tfrt.chain)

  tfrt.return
}
