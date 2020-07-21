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

// RUN: bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running 'BM_Tf_BiasAdd_1x1001_f32'
func @BM_Tf_BiasAdd_1x1001_f32() {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 1001], values = [1.0 : f32] } : 1
  %bias = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1001], values = [1.0 : f32] } : 1

  tfrt_test.benchmark "BM_Tf_BiasAdd_1x1001_f32"(
      %gpu      : !corert.device,
      %input    : !corert.tensorhandle,
      %bias     : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
    %result = corert.executeop(%gpu) "tf.BiasAdd"(%input, %bias) { T = f32 } : 1
    %done = corert.executeop(%gpu) "tfrt_test.synchronize"(%result) : 1
    tfrt.return %done : !corert.tensorhandle
  }

  tfrt.return
}
