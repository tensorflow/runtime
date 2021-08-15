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

// RUN: bef_executor %s.bef --work_queue_type=mstd:8 | FileCheck %s

// Layer (type)                 Output Shape              Param #
// =================================================================
// flatten_1 (Flatten)          (None, 784)               0
// _________________________________________________________________
// dense (Dense)                (None, 128)               100480
// _________________________________________________________________
// dense_1 (Dense)              (None, 10)                1290
// =================================================================
// Total params: 101,770
// Trainable params: 101,770
// Non-trainable params: 0

// CHECK-LABEL: --- Running 'mnist_training'
func @mnist_training() {
  %path = "tfrt_test.get_string"() { value = "integrationtest/mnist/test_data/mnist_metadata.btf" } : () -> !tfrt.string
  %c0 = tfrt.new.chain

  // Initialize Variables (Weights, Bias, Input images, etc.).
  %train_image_index = tfrt.constant.i32 0
  %train_label_onehot_index = tfrt.constant.i32 1
  %w0_index = tfrt.constant.i32 2
  %b0_index = tfrt.constant.i32 3
  %w1_index = tfrt.constant.i32 4
  %b1_index = tfrt.constant.i32 5
  %expected_result_index = tfrt.constant.i32 6

  %train_image = "btf.read_dense_tensor.f32.2"(%path, %train_image_index) : (!tfrt.string, i32) -> (!t.tensor)
  %train_label_onehot = "btf.read_dense_tensor.f32.2"(%path, %train_label_onehot_index) : (!tfrt.string, i32) -> (!t.tensor)
  // W_0 shape: (784, 512)
  %w_0 = "btf.read_dense_tensor.f32.2"(%path, %w0_index) : (!tfrt.string, i32) -> (!t.tensor)
  // b_0 shape: (512,)
  %b_0 = "btf.read_dense_tensor.f32.1"(%path, %b0_index) : (!tfrt.string, i32) -> (!t.tensor)
  // W_1 shape: (512, 10)
  %w_1 = "btf.read_dense_tensor.f32.2"(%path, %w1_index) : (!tfrt.string, i32) -> (!t.tensor)
  // b_1 shape: (10,)
  %b_1 = "btf.read_dense_tensor.f32.1"(%path, %b1_index) : (!tfrt.string, i32) -> (!t.tensor)
  %expected_result = "btf.read_dense_tensor.f32.2"(%path, %expected_result_index) : (!tfrt.string, i32) -> (!t.tensor)

  %step_num = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %max_step_num = tfrt.constant.i32 11
  // Learning rate: 0.01
  %lr = "tfrt_dht.create_uninitialized_tensor.f32.1"() { shape = [1: i64] } : () -> !t.tensor
  %c1 = tfrt_dht.fill_tensor_with_constant.f32 %lr, %c0 0.01 : f32
  // Used in updating W.
  %minus_lr_constant = tfrt.constant.f32 -0.01

  // Gradient buffers
  %gradient = "tfrt_dht.create_uninitialized_tensor.f32.2"() { shape = [1: i64, 10 : i64] } : () -> !t.tensor
  %gradient_a0 = "tfrt_dht.create_uninitialized_tensor.f32.2"() { shape = [1: i64, 512 : i64] } : () -> !t.tensor
  %gradient_b1 = "tfrt_dht.create_uninitialized_tensor.f32.1"() { shape = [10 : i64] } : () -> !t.tensor
  %gradient_b0 = "tfrt_dht.create_uninitialized_tensor.f32.1"() { shape = [512 : i64] } : () -> !t.tensor

  %transposed_activation_0 = "tfrt_dht.create_uninitialized_tensor.f32.2"() { shape = [512 : i64, 1 : i64] } : () -> !t.tensor
  %transposed_w_1 = "tfrt_dht.create_uninitialized_tensor.f32.2"() { shape = [10 : i64, 512 :i64] } : () -> !t.tensor
  %transposed_input_image = "tfrt_dht.create_uninitialized_tensor.f32.2"() { shape = [784 : i64, 1 : i64] } : () -> !t.tensor

  // Other constants
  %one = tfrt.constant.f32 1.0
  %zero = tfrt.constant.f32 0.0

  // Use benchmark_kernels as training loop driver.
  tfrt_test.benchmark "mnist_training_benchmark"(
    %c0 : !tfrt.chain,
    %c1 : !tfrt.chain,

    %train_image : !t.tensor,
    %train_label_onehot : !t.tensor,

    // Trainable variables.
    %w_0 : !t.tensor,
    %b_0 : !t.tensor,
    %w_1 : !t.tensor,
    %b_1 : !t.tensor,

    // Gradient buffers.
    %gradient : !t.tensor,
    %gradient_a0 : !t.tensor,
    %gradient_b1 : !t.tensor,
    %gradient_b0 : !t.tensor,
    %transposed_activation_0 : !t.tensor,
    %transposed_w_1 : !t.tensor,
    %transposed_input_image : !t.tensor,

    %lr : !t.tensor,

    %expected_result : !t.tensor,
    %one : f32,
    %zero : f32,
    %minus_lr_constant: f32,
    %step_num : !test.atomic.i32,
    %max_step_num : i32
  )
  duration_secs = 100, max_count = 11, num_warmup_runs = 0
  {
    ////////////////////////////////////////
    // Forward Pass.
    ////////////////////////////////////////
    // y0 = w0 * Input + b0
    %shape_b0 = ts.build_shape [1 : i64, 512 : i64]
    %activation_0 = "tfrt_test.broadcast.f32.2"(%b_0, %shape_b0, %c0) : (!t.tensor, !ts.shape, !tfrt.chain) -> !t.tensor
    %c2 = "eigen.matmul.f32"(%one, %train_image, %w_0, %one, %activation_0, %c0) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c3 = tfrt.merge.chains %c1, %c2 : !tfrt.chain, !tfrt.chain
    // a0 = relu(y0)
    %c4 = "tfrt_test.relu_inplace.f32"(%activation_0, %c3) : (!t.tensor, !tfrt.chain) -> !tfrt.chain

    // y1 = w1 * y0 + b1
    %shape_b1 = ts.build_shape [1: i64, 10: i64]
    %activation_1 = "tfrt_test.broadcast.f32.2"(%b_1, %shape_b1, %c4) : (!t.tensor, !ts.shape, !tfrt.chain) -> !t.tensor
    %c5 = "eigen.matmul.f32"(%one, %activation_0, %w_1, %one, %activation_1, %c0) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // y1 = softmax(y1)
    %c6 = "tfrt_test.softmax_inplace.f32"(%activation_1, %c5) : (!t.tensor, !tfrt.chain) -> !tfrt.chain

    ////////////////////////////////////////
    // Increment step_num and result check.
    ////////////////////////////////////////
    %c27 = "tfrt_test.atomic.inc.i32"(%step_num, %c0) : (!test.atomic.i32, !tfrt.chain) -> !tfrt.chain
    %step_num_val, %c28 = "tfrt_test.atomic.get.i32"(%step_num, %c27) : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)
    %cond = "tfrt.lessequal.i32"(%max_step_num, %step_num_val) : (i32, i32) -> (i1)
    // Check prediction result matching the one generated from current
    // TensorFlow, after training for 10 steps.
    tfrt.if %cond, %activation_1, %expected_result, %c6 : (!t.tensor, !t.tensor, !tfrt.chain) -> () {
      %cmp, %c28 = "tfrt_dht.tensor_allclose.1000ulp.f32"(%expected_result, %activation_1, %c6) : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)
      // CHECK: int1 = 1
      %c29 = "tfrt.print.i1"(%cmp, %c28) : (i1, !tfrt.chain) -> !tfrt.chain
      %c30 = tfrt_dht.print_tensor %activation_1, %c29
      %c31 = tfrt_dht.print_tensor %expected_result, %c30
      tfrt.return
    }

    ////////////////////////////////////////
    // Backward Pass.
    ////////////////////////////////////////
    %c7 = "tfrt_test.flatten.f32"(%activation_1, %gradient, %c6) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c8 = "tfrt_test.subtract_inplace.f32"(%train_label_onehot, %gradient, %c7) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    // Get the mean for minibatch SGD
    %c9 = "tfrt_test.mean_axis_zero.f32"(%gradient, %gradient_b1, %c8) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c10 = tfrt.merge.chains %c1, %c9 : !tfrt.chain, !tfrt.chain
    // Update b_1.
    %c11 = "tfrt_test.gradient_descent.f32"(%gradient_b1, %lr, %b_1, %c10) : (!t.tensor, !t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // Update W_1. Gradient descent implemented with GEMM: C = alpha * A * B + beta * C
    // Reference: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    %c12 = "tfrt_test.tensor_transpose.f32"(%activation_0, %transposed_activation_0, %c4) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // Calculate transpose of W_1 before W_1 gets updated.
    %c14 = "tfrt_test.tensor_transpose.f32"(%w_1, %transposed_w_1, %c0) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c15 = tfrt.merge.chains %c8, %c12, %c14 : !tfrt.chain, !tfrt.chain, !tfrt.chain

    // Update W_1.
    %c16 = "eigen.matmul.f32"(%minus_lr_constant, %transposed_activation_0, %gradient, %one, %w_1, %c15) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // Calculate gradient(a0).
    %c17 = "eigen.matmul.f32"(%one, %gradient, %transposed_w_1, %zero, %gradient_a0, %c15) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c18 = tfrt.merge.chains %c4, %c17 : !tfrt.chain, !tfrt.chain
    %c19 = "tfrt_test.relu_grad_inplace.f32"(%activation_0, %gradient_a0, %c18) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // Update b_0.
    %c20 = "tfrt_test.mean_axis_zero.f32"(%gradient_a0, %gradient_b0, %c19) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c21 = tfrt.merge.chains %c1, %c20 : !tfrt.chain, !tfrt.chain
    %c22 = "tfrt_test.gradient_descent.f32"(%gradient_b0, %lr, %b_0, %c21) : (!t.tensor, !t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

    // Update W_0.
    %c23 = "tfrt_test.tensor_transpose.f32"(%train_image, %transposed_input_image, %c0) : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain
    %c24 = tfrt.merge.chains %c19, %c23 : !tfrt.chain, !tfrt.chain
    %c25 = "eigen.matmul.f32"(%minus_lr_constant, %transposed_input_image, %gradient_a0, %one, %w_0, %c24) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !tfrt.chain) -> !tfrt.chain

    %c_done = tfrt.merge.chains %c28, %c11, %c16, %c22, %c25 : !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain
    tfrt.return %c_done : !tfrt.chain
  }

  tfrt.return
}
