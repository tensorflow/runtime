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

func @mnist_compute(%w1 : !t.tensor,
                    %b1 : !t.tensor,
                    %w2 : !t.tensor,
                    %b2 : !t.tensor,
                    %test_input_features : !t.tensor,
                    %test_input_labels : !t.tensor,
                    %ch0 : !tfrt.chain) -> !t.tensor
{
  %one = tfrt.constant.f32 1.0

  %target_shape_b1 = ts.build_shape [100 : i64, 512 : i64]
  // Shape: [100, 512].
  %activation1 = "tfrt_test.broadcast.f32.2"(%b1, %target_shape_b1, %ch0) : (!t.tensor, !ts.shape, !tfrt.chain) -> !t.tensor
  // Shape: [100, 512].
  %ch1 = "tfrt_test.matmul.f32.2"(%one, %test_input_features, %w1, %one, %activation1, %ch0) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain
  // Shape: [100, 512].
  %relu1_ch = "tfrt_test.relu_inplace.f32"(%activation1, %ch1) : (!t.tensor, !tfrt.chain) -> !tfrt.chain

  %target_shape_b2 = ts.build_shape [100 : i64, 10 : i64]
  // Shape: [100, 10].
  %activation2 = "tfrt_test.broadcast.f32.2"(%b2, %target_shape_b2, %ch0) : (!t.tensor, !ts.shape, !tfrt.chain) ->  !t.tensor
  // Shape: [100, 10].
  %ch2 = "tfrt_test.matmul.f32.2"(%one, %activation1, %w2, %one, %activation2, %relu1_ch) :  (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain

  // Shape: [100].
  %argmax_h2 = "tfrt_test.argmax.f32.2"(%activation2, %ch2) : (!t.tensor, !tfrt.chain) ->  !t.tensor
  // Shape: [100].
  %equal_ch = "tfrt_test.equal_inplace.i32"(%test_input_labels, %argmax_h2, %ch2): (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
  %equal_f32 = "tfrt_dht.create_uninitialized_tensor.f32.1"() { shape = [100 : i64] } :
    () -> !t.tensor
  %cast_ch = "tfrt_test.cast.i32_to_f32"(%argmax_h2, %equal_f32, %equal_ch): (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
  // Shape: [].
  %avg_accuracy = "tfrt_test.reduce_mean.f32.1"(%equal_f32, %cast_ch): (!t.tensor, !tfrt.chain) -> !t.tensor
  tfrt.return %avg_accuracy : !t.tensor
}


// CHECK-LABEL: --- Running 'mnist'
func @mnist() -> !tfrt.chain {
  %path = "tfrt_test.get_string"() { value = "integrationtest/mnist/test_data/mnist_tensors.btf" } : () -> !tfrt.string
  %w1_index = tfrt.constant.i32 0
  %b1_index = tfrt.constant.i32 1
  %w2_index = tfrt.constant.i32 2
  %b2_index = tfrt.constant.i32 3
  %test_input_features_index = tfrt.constant.i32 4
  %test_input_labels_index = tfrt.constant.i32 5
  // Shape: [784, 512].
  %w1 = "btf.read_dense_tensor.f32.2"(%path, %w1_index) : (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [512].
  %b1 = "btf.read_dense_tensor.f32.1"(%path, %b1_index) : (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [512, 10].
  %w2 = "btf.read_dense_tensor.f32.2"(%path, %w2_index) : (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [10].
  %b2 = "btf.read_dense_tensor.f32.1"(%path, %b2_index) : (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [100, 784].
  %test_input_features = "btf.read_dense_tensor.f32.2"(%path, %test_input_features_index) : (!tfrt.string, i32) -> (!t.tensor)
  // Shape: [100].
  %test_input_labels = "btf.read_dense_tensor.i32.1"(%path, %test_input_labels_index) : (!tfrt.string, i32) -> (!t.tensor)
  %c = tfrt.new.chain

  %avg_accuracy = tfrt.call @mnist_compute(%w1, %b1, %w2, %b2, %test_input_features, %test_input_labels, %c)
       : (!t.tensor, !t.tensor, !t.tensor,
          !t.tensor, !t.tensor, !t.tensor, !tfrt.chain)
       -> !t.tensor

  // CHECK: shape = [], values = [9.800000e-01]
  %c1 = tfrt_dht.print_tensor %avg_accuracy, %c

  tfrt.return %c1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'bm_mnist'
func @bm_mnist() {
  %ch0 = tfrt.new.chain
  %w1 = tfrt_dht.create_uninitialized_tensor.f32.2 [784 : i64, 512 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %w1, %ch0 1.0 : f32
  %b1 = tfrt_dht.create_uninitialized_tensor.f32.1 [512 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %b1, %ch0 1.0 : f32
  %w2 = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 10 : i64]
  %ch3 = tfrt_dht.fill_tensor_with_constant.f32 %w2, %ch0 1.0 : f32
  %b2 = tfrt_dht.create_uninitialized_tensor.f32.1 [10 : i64]
  %ch4 = tfrt_dht.fill_tensor_with_constant.f32 %b2, %ch0 1.0 : f32
  %test_input_features = tfrt_dht.create_uninitialized_tensor.f32.2 [100 : i64, 784 : i64]
  %ch5 = tfrt_dht.fill_tensor_with_constant.f32 %test_input_features, %ch0 1.0 : f32
  %test_input_labels = tfrt_dht.create_uninitialized_tensor.i32.1 [100 : i64]
  %ch6 = tfrt_dht.fill_tensor_with_constant.i32 %test_input_labels, %ch0 7 : i32
  %ch7 = tfrt.merge.chains %ch1, %ch2, %ch3, %ch4, %ch5, %ch6 : !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain

  tfrt_test.benchmark "bm_mnist"(
      %w1 : !t.tensor,
      %b1 : !t.tensor,
      %w2 : !t.tensor,
      %b2 : !t.tensor,
      %test_input_features : !t.tensor,
      %test_input_labels : !t.tensor,
      %ch7 : !tfrt.chain)
      duration_secs = 10, max_count = 10000, num_warmup_runs = 10 {
      %avg_accuracy = tfrt.call @mnist_compute(%w1, %b1, %w2, %b2, %test_input_features, %test_input_labels, %ch7)
       : (!t.tensor, !t.tensor,
          !t.tensor, !t.tensor,
          !t.tensor, !t.tensor,
          !tfrt.chain) -> !t.tensor

      tfrt.return %avg_accuracy : !t.tensor
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_broadcast'
func @test_broadcast() {
  %ch0 = tfrt.new.chain
  %tensor_1 = "tfrt_dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64] } : () -> !t.tensor
  %ch1 = "tfrt_dht.set_tensor_with_constant_values.f32"(%tensor_1, %ch0)
    { values = [1.0 : f32, 2.0 : f32] } : (!t.tensor, !tfrt.chain) -> !tfrt.chain
  %target_shape = ts.build_shape [3: i64, 1 : i64, 2 : i64]
  %tensor_2 = "tfrt_test.broadcast.f32.3"(%tensor_1, %target_shape, %ch1)
    : (!t.tensor, !ts.shape, !tfrt.chain) -> !t.tensor

  // CHECK-NEXT: DenseHostTensor dtype = f32, shape = [3, 1, 2], values = [1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00]
  %ch2 = tfrt_dht.print_tensor %tensor_2, %ch0

  tfrt.return
}
