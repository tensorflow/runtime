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

// RUN: bef_executor %s.bef | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'test_flatten_f32'
func @test_flatten_f32() {
  %ch0 = tfrt.new.chain

  %t1 = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 2 : i64, 2 : i64, 1 : i64] }
    : () -> !t.tensor
  %ch1 = "tfrt_dht.set_tensor_with_constant_values.f32"(%t1, %ch0)
    { values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] }
    : (!t.tensor, !tfrt.chain) -> !tfrt.chain

  %t2 = "tfrt_dht.create_uninitialized_tensor.f32.2"()
    { shape = [1 : i64, 4 : i64] }
    : () -> !t.tensor
  %ch2 = "tfrt_test.flatten.f32"(%t1, %t2, %ch1)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [1, 4], values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]
  tfrt_dht.print_tensor %t2, %ch2
  tfrt.return
}


// CHECK-LABEL: --- Running 'test_max_pool_2d_f32_padding_error'
func @test_max_pool_2d_f32_padding_error() {
  %ch0 = tfrt.new.chain

  %input = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 3 : i64, 3 : i64, 6 : i64] }
    : () -> !t.tensor
  %output = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 1 : i64, 1 : i64, 6 : i64] }
    : () -> !t.tensor

  // expected-error @+1 {{padding 'invalid' is not recognized}}
  "tfrt_test.max_pooling_2d.f32"(%input, %output, %ch0)
    { padding = "invalid", pool_size = [3 : i32, 3 : i32], strides = [3 : i32, 3 : i32] }
    :  (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_max_pool_2d_f32_shape_error'
func @test_max_pool_2d_f32_shape_error() {
  %ch0 = tfrt.new.chain

  %input = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 3 : i64, 3 : i64, 6 : i64] }
    : () -> !t.tensor
  %output = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 2 : i64, 1 : i64, 6 : i64] }
    : () -> !t.tensor

  // expected-error @+1 {{output shape [2, 2, 1, 6] does not match the expected output shape [2, 1, 1, 6]}}
  "tfrt_test.max_pooling_2d.f32"(%input, %output, %ch0)
    { padding = "valid", pool_size = [3 : i32, 3 : i32], strides = [3 : i32, 3 : i32] }
    :  (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
