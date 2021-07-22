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

// CHECK-LABEL: --- Running 'BM_Conv2DGradFilter_in_8x32x32x128_f_1x1x128_s1x1_SAME'
func @BM_Conv2DGradFilter_in_8x32x32x128_f_1x1x128_s1x1_SAME() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // in: [8, 32, 32, 128].
  %in = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // output gradient: [8, 32, 32, 128].
  %out_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %out_grad, %ch1 1.0 : f32

  // filter gradient: [1, 1, 128, 128].
  %filter_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 1 : i64, 128 : i64, 128 : i64] }
    : () -> !t.tensor

  tfrt_test.benchmark "BM_Conv2DGradFilter_in_8x32x32x128_f_1x1x128_s1x1_SAME"(
      %in          : !t.tensor,
      %out_grad    : !t.tensor,
      %filter_grad : !t.tensor,
      %ch2         : !tfrt.chain
  )
  duration_secs = 5, max_count = 250, num_warmup_runs = 50
  {
      %ch_out = "eigen.conv2d.grad.filter.f32"(%out_grad, %in, %filter_grad, %ch2)
       { padding = "same",  strides = [1 : i64, 1 : i64] }
       : (!t.tensor, !t.tensor, !t.tensor,
          !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_Conv2DGradFilter_in_8x56x56x128_f_2x2x128_s2x2_SAME'
func @BM_Conv2DGradFilter_in_8x56x56x128_f_2x2x128_s2x2_SAME() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // in: [8, 56, 56, 128].
  %in = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 56 : i64, 56 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // output gradient: [8, 28, 28, 128].
  %out_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 28 : i64, 28 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %out_grad, %ch1 1.0 : f32

  // filter gradient: [2, 2, 128, 128].
  %filter_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 2 : i64, 128 : i64, 128 : i64] }
    : () -> !t.tensor

  tfrt_test.benchmark "BM_Conv2DGradFilter_in_8x56x56x128_f_2x2x128_s2x2_SAME"(
      %in          : !t.tensor,
      %out_grad    : !t.tensor,
      %filter_grad : !t.tensor,
      %ch2         : !tfrt.chain
  )
  duration_secs = 5, max_count = 250, num_warmup_runs = 50
  {
      %ch_out = "eigen.conv2d.grad.filter.f32"(%out_grad, %in, %filter_grad, %ch2)
       { padding = "same",  strides = [2 : i64, 2 : i64] }
       : (!t.tensor, !t.tensor, !t.tensor,
          !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_Conv2DGradFilter_in_8x28x28x256_f_2x2x256_s2x2_SAME'
func @BM_Conv2DGradFilter_in_8x28x28x256_f_2x2x256_s2x2_SAME() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // in: [8, 28, 28, 256].
  %in = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 28 : i64, 28 : i64, 256 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // output gradient: [8, 14, 14, 256].
  %out_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 14 : i64, 14 : i64, 256 : i64] }
    : () -> !t.tensor
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %out_grad, %ch1 1.0 : f32

  // filter gradient: [2, 2, 256, 256].
  %filter_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 2 : i64, 256 : i64, 256 : i64] }
    : () -> !t.tensor

  tfrt_test.benchmark "BM_Conv2DGradFilter_in_8x28x28x256_f_2x2x256_s2x2_SAME"(
      %in          : !t.tensor,
      %out_grad    : !t.tensor,
      %filter_grad : !t.tensor,
      %ch2         : !tfrt.chain
  )
  duration_secs = 5, max_count = 250, num_warmup_runs = 50
  {
      %ch_out = "eigen.conv2d.grad.filter.f32"(%out_grad, %in, %filter_grad, %ch2)
       { padding = "same",  strides = [2 : i64, 2 : i64] }
       : (!t.tensor, !t.tensor, !t.tensor,
          !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}
