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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'BM_Conv2D_in_8x32x32x128_f_1x1x128'
func @BM_Conv2D_in_8x32x32x128_f_1x1x128() {
  %ch0 = hex.new.chain

  %zero = hex.constant.f32 0.0
  %one = hex.constant.f32 1.0

  // in: [8, 32, 32, 128].
  %in = "dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch1 = dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // kern: [1, 1, 128, 128].
  %kern = "dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 1 : i64, 128 : i64, 128 : i64] }
    : () -> !t.tensor
  %ch2 = dht.fill_tensor_with_constant.f32 %kern, %ch1 1.0 : f32

  // scale: [128].
  %scale = "dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch3 = dht.fill_tensor_with_constant.f32 %scale, %ch2 1.0 : f32

  // offset: [128].
  %offset = "dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch4 = dht.fill_tensor_with_constant.f32 %offset, %ch2 1.0 : f32

  // mean: [128].
  %mean = "dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch5 = dht.fill_tensor_with_constant.f32 %mean, %ch2 1.0 : f32

  // var: [128].
  %var = "dht.create_uninitialized_tensor.f32.1"()
    { shape = [128 : i64] }
    : () -> !t.tensor
  %ch6 = dht.fill_tensor_with_constant.f32 %var, %ch2 1.0 : f32

  // out: [8, 32, 32, 128].
  %out = "dht.create_uninitialized_tensor.f32.4"()
    { shape = [8 : i64, 32 : i64, 32 : i64, 128 : i64] }
    : () -> !t.tensor

      %in     : !t.tensor,
      %kern   : !t.tensor,
      %scale  : !t.tensor,
      %offset : !t.tensor,
      %mean   : !t.tensor,
      %var    : !t.tensor,
      %out    : !t.tensor,
      %ch6    : !hex.chain
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
      %ch_out = "eigen.conv2d.batch_norm.f32"(%in, %kern, %scale, %offset,
                                              %mean, %var, %out, %ch6)
       { epsilon = 0.01 : f32, padding = "valid", strides = [1 : i64, 1 : i64] }
       : (!t.tensor, !t.tensor, !t.tensor, !t.tensor, !t.tensor, !t.tensor,
          !t.tensor, !hex.chain) -> !hex.chain

      hex.return %ch_out : !hex.chain
  }

  hex.return
}
