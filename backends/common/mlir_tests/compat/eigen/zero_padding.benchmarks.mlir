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

// CHECK-LABEL: --- Running 'BM_ZeroPadding_in_1x224x224x3_p_3x3'
func @BM_ZeroPadding_in_1x224x224x3_p_3x3() {
  %ch0 = tfrt.new.chain

  %zero = tfrt.constant.f32 0.0
  %one = tfrt.constant.f32 1.0

  // in: [1, 224, 224, 3].
  %in = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 224 : i64, 224 : i64, 3 : i64] }
    : () -> !t.tensor
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %in, %ch0 1.0 : f32

  // out: [1, 230, 230, 3].
  %out = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [1 : i64, 230 : i64, 230 : i64, 3 : i64] }
    : () -> !t.tensor

  tfrt_test.benchmark "BM_ZeroPadding_in_1x224x224x3_p_3x3"(
      %in   : !t.tensor,
      %out  : !t.tensor,
      %ch1  : !tfrt.chain
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
      %ch_out = "eigen.zero_padding.f32"(%in, %out, %ch1)
       { padding = [3 : i64, 3 : i64] }
       : (!t.tensor, !t.tensor, !tfrt.chain) -> !tfrt.chain

      tfrt.return %ch_out : !tfrt.chain
  }

  tfrt.return
}
