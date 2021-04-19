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

// CHECK-LABEL: --- Running 'BM_MatMul_512x512x512_f32'
func @BM_MatMul_512x512x512_f32() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // Shape: [512, 512].
  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %ch0 1.0 : f32

  // Shape: [512, 512].
  %b = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %b, %ch0 1.0 : f32

  // Shape: [512, 512].
  %c = tfrt_dht.create_uninitialized_tensor.f32.2 [512 : i64, 512 : i64]
  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain

  tfrt_test.benchmark "BM_MatMul_512x512x512_f32"(
      %zero : f32,
      %one : f32,
      %a : !t.tensor,
      %b : !t.tensor,
      %c : !t.tensor,
      %ch3 : !tfrt.chain)
  duration_secs = 5, max_count = 1000, num_warmup_runs = 50
  {
      %ch_out = "eigen.matmul.f32"(%one, %a, %b, %zero, %c, %ch3)
       : (f32, !t.tensor, !t.tensor, f32,
          !t.tensor, !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_MatMul_1024x1024x1024_f32'
func @BM_MatMul_1024x1024x1024_f32() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // Shape: [1024, 1024].
  %a = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %a, %ch0 1.0 : f32

  // Shape: [1024, 1024].
  %b = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %b, %ch0 1.0 : f32

  // Shape: [1024, 1024].
  %c = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain

  tfrt_test.benchmark "BM_MatMul_1024x1024x1024_f32"(
      %zero : f32,
      %one : f32,
      %a : !t.tensor,
      %b : !t.tensor,
      %c : !t.tensor,
      %ch3 : !tfrt.chain)
  duration_secs = 5, max_count = 1000, num_warmup_runs = 50
  {
      %ch_out = "eigen.matmul.f32"(%one, %a, %b, %zero, %c, %ch3)
       : (f32, !t.tensor, !t.tensor, f32,
          !t.tensor, !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}
