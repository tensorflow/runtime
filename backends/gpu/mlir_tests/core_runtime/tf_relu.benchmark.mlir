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

// RUN: bef_executor --test_init_function=register_op_handlers_gpu %s.bef | FileCheck %s --dump-input=always

func @register_op_handlers_gpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %gpu_ordinal = tfrt.constant.i32 0
  %gpu = "corert.create_gpu_op_handler" (%gpu_ordinal, %null) : (i32, !corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %gpu "gpu"
  tfrt.return
}

// CHECK: --- Running 'BM_Tf_Relu_1x56x56x256_f32'
func @BM_Tf_Relu_1x56x56x256_f32() {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 56, 56, 256], values = [1.0 : f32] } : 1

  tfrt_test.benchmark "BM_Tf_Relu_1x56x56x256_f32"(
      %gpu      : !corert.ophandler,
      %input    : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
    %result = corert.executeop(%gpu) "tf.Relu"(%input) { T = f32 } : 1
    %done = corert.executeop(%gpu) "tfrt_test.synchronize"(%result) : 1
    tfrt.return %done : !corert.tensorhandle
  }

  tfrt.return
}
