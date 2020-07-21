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

// CHECK: --- Running 'BM_Tf_Conv2d_1x256x56x56_3x3_256_f32'
func @BM_Tf_Conv2d_1x256x56x56_3x3_256_f32() {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 256, 56, 56], values = [1.0 : f32] } : 1

  %filter = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [3, 3, 256, 256], values = [1.0 : f32] } : 1

  tfrt_test.benchmark "BM_Tf_Conv2d_1x256x56x56_3x3_256_f32"(
      %gpu      : !corert.device,
      %input    : !corert.tensorhandle,
      %filter   : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
    %result = corert.executeop(%gpu)
      "tf.Conv2D"(%input, %filter)
      {
        T = f32,
        data_format= "NCHW",
        padding = "VALID",
        strides = [1, 1, 1, 1],
        dilations = [1, 1, 1, 1]
      } : 1

    %done = corert.executeop(%gpu) "tfrt_test.synchronize"(%result) : 1

    tfrt.return %done : !corert.tensorhandle
  }

  tfrt.return
}

// CHECK: --- Running 'BM_Tf_Conv2d_1x56x56x256_3x3_256_f16'
func @BM_Tf_Conv2d_1x56x56x256_3x3_256_f16() {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %input_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 56, 56, 256], values = [1.0 : f32] } : 1

  %filter_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [3, 3, 256, 256], values = [1.0 : f32] } : 1

  %input = corert.executeop(%gpu) "tf.Cast"(%input_f32)
    { DstT = f16, SrcT = f32, Truncate = true } : 1

  %filter = corert.executeop(%gpu) "tf.Cast"(%filter_f32)
    { DstT = f16, SrcT = f32, Truncate = true } : 1

  tfrt_test.benchmark "BM_Tf_Conv2d_1x56x56x256_3x3_256_f16"(
      %gpu      : !corert.device,
      %input    : !corert.tensorhandle,
      %filter   : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
    %result = corert.executeop(%gpu)
      "tf.Conv2D"(%input, %filter)
      {
        T = f16,
        data_format= "NHWC",
        padding = "VALID",
        strides = [1, 1, 1, 1],
        dilations = [1, 1, 1, 1]
      } : 1

    %done = corert.executeop(%gpu) "tfrt_test.synchronize"(%result) : 1

    tfrt.return %done : !corert.tensorhandle
  }

  tfrt.return
}

// CHECK: --- Running 'BM_Tf_Conv2d_1x56x56x256_1x1_256_f16'
func @BM_Tf_Conv2d_1x56x56x256_1x1_256_f16() {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %input_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 56, 56, 256], values = [1.0 : f32] } : 1

  %filter_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 1, 256, 256], values = [1.0 : f32] } : 1

  %input = corert.executeop(%gpu) "tf.Cast"(%input_f32)
    { DstT = f16, SrcT = f32, Truncate = true } : 1

  %filter = corert.executeop(%gpu) "tf.Cast"(%filter_f32)
    { DstT = f16, SrcT = f32, Truncate = true } : 1

  tfrt_test.benchmark "BM_Tf_Conv2d_1x56x56x256_1x1_256_f16"(
      %gpu      : !corert.device,
      %input    : !corert.tensorhandle,
      %filter   : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000, num_warmup_runs = 10
  {
    %result = corert.executeop(%gpu)
      "tf.Conv2D"(%input, %filter)
      {
        T = f16,
        data_format= "NHWC",
        padding = "VALID",
        strides = [1, 1, 1, 1],
        dilations = [1, 1, 1, 1]
      } : 1

    %done = corert.executeop(%gpu) "tfrt_test.synchronize"(%result) : 1

    tfrt.return %done : !corert.tensorhandle
  }

  tfrt.return
}
