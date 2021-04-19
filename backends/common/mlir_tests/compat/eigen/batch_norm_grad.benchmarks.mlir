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

// RUN: bef_executor $(bef_name %s) | FileCheck %s

// CHECK-LABEL: --- Running 'BM_BatchNormGrad_8x32x32x128'
func @BM_BatchNormGrad_8x32x32x128() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // output_grad: [8, 32, 32, 128]
  %output_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %output_grad, %ch0 3.0 : f32

  // input: [8, 32, 32, 128]
  %input = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 2.0 : f32

  // gamma: [128]
  %gamma = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch3 = tfrt_dht.fill_tensor_with_constant.f32 %gamma, %ch0 0.25 : f32

  // mean: [128]
  %mean = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch4 = tfrt_dht.fill_tensor_with_constant.f32 %mean, %ch0 0.25 : f32

  // variance: [128]
  %variance = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch5 = tfrt_dht.fill_tensor_with_constant.f32 %variance, %ch0 0.25 : f32

  // input_grad: [8, 32, 32, 128]
  %input_grad = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor

  // gamma_grad: [128]
  %gamma_grad = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor

  // beta_grad: [128]
  %beta_grad = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor

  %init = tfrt.merge.chains %ch0, %ch1, %ch2, %ch3, %ch4, %ch5 : !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain, !tfrt.chain

  tfrt_test.benchmark "BM_BatchNormGrad_8x32x32x128"(
      %output_grad : !t.tensor,
      %input       : !t.tensor,
      %gamma       : !t.tensor,
      %mean        : !t.tensor,
      %variance    : !t.tensor,
      %input_grad  : !t.tensor,
      %gamma_grad  : !t.tensor,
      %beta_grad   : !t.tensor,
      %init        : !tfrt.chain
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
      %ch_in, %ch_gamma, %ch_beta = "eigen.batch_norm.grad.f32"
       (
        %output_grad, %input, %gamma, %mean, %variance, %init, %input_grad,
        %gamma_grad, %beta_grad
       )
       { epsilon = 0.01 : f32 }
       : (!t.tensor, !t.tensor, !t.tensor, !t.tensor, !t.tensor, !tfrt.chain,
          !t.tensor, !t.tensor, !t.tensor
         ) -> (!tfrt.chain, !tfrt.chain, !tfrt.chain)

      %done = tfrt.merge.chains %ch_in, %ch_gamma, %ch_beta : !tfrt.chain, !tfrt.chain, !tfrt.chain
      tfrt.return %done : !tfrt.chain
  }

  tfrt.return
}
