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

// CHECK-LABEL: --- Running 'BM_Conv2D_in_8x32x32x128_f_1x1x128'
func @BM_Conv2D_in_8x32x32x128_f_1x1x128() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // in: [8, 32, 32, 128].
  %in = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // kern: [1, 1, 128, 128].
  %kern = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 1 : i64, 128 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %kern, %ch1 1.0 : f32

  // scale: [128].
  %scale = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch3 = tfrt_dht.fill_tensor_with_constant.f32 %scale, %ch2 1.0 : f32

  // offset: [128].
  %offset = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch4 = tfrt_dht.fill_tensor_with_constant.f32 %offset, %ch2 1.0 : f32

  // mean: [128].
  %mean = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch5 = tfrt_dht.fill_tensor_with_constant.f32 %mean, %ch2 1.0 : f32

  // var: [128].
  %var = "tfrt_dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch6 = tfrt_dht.fill_tensor_with_constant.f32 %var, %ch2 1.0 : f32

  // out: [8, 32, 32, 128].
  %out = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  tfrt_test.benchmark "BM_Conv2D_in_8x32x32x128_f_1x1x128"(
      %in     : !t.tensor,
      %kern   : !t.tensor,
      %scale  : !t.tensor,
      %offset : !t.tensor,
      %mean   : !t.tensor,
      %var    : !t.tensor,
      %out    : !t.tensor,
      %ch6    : !tfrt.chain
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
      %ch_out = "eigen.conv2d.batch_norm.f32"(%in, %kern, %scale, %offset,
                                              %mean, %var, %out, %ch6)
       { epsilon = 0.01 : f32, padding = "valid", strides = [1 : i64, 1 : i64] }
       : (!t.tensor, !t.tensor, !t.tensor, !t.tensor, !t.tensor, !t.tensor,
          !t.tensor, !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}
