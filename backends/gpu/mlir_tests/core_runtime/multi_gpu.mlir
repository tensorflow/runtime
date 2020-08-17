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

// RUN: bef_executor -devices=cpu $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running '__init__'
func @__init__() -> !tfrt.chain {
  %null = "corert.create_null_op_handler"() : () -> !corert.device
  %ordinal_0 = tfrt.constant.i32 0
  %gpu_0 = "corert.create_gpu_op_handler" (%ordinal_0, %null) : (i32, !corert.device) -> !corert.device
  %ch_0 = corert.register_op_handler %gpu_0 "gpu0"

  %ordinal_1 = tfrt.constant.i32 1
  %gpu_1 = "corert.create_gpu_op_handler" (%ordinal_1, %null) : (i32, !corert.device) -> !corert.device
  %ch_1 = corert.register_op_handler %gpu_1 "gpu1"
  tfrt.return %ch_1 : !tfrt.chain
}


// CHECK: --- Running 'relu'
func @relu() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu0 = corert.get_op_handler %ch0 "gpu0"
  %gpu1 = corert.get_op_handler %ch0 "gpu1"

  %gpu_handle_input0 = corert.executeop(%gpu0)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1

  %gpu_handle_result0 = corert.executeop(%gpu0) "tf.Relu"(%gpu_handle_input0) : 1
  %ch_print_gpu0 = corert.executeop.seq(%gpu0, %ch0) "tfrt_test.print"(%gpu_handle_result0) : 0

  %cpu_handle_result0 = corert.executeop(%gpu0) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 5], values = [0, 0, 0, 0.5, 1]
  %ch_print_cpu0 = corert.executeop.seq(%gpu0, %ch0) "tfrt_test.print"(%cpu_handle_result0) : 0


  %gpu_handle_input1 = corert.executeop(%gpu1)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1

  %gpu_handle_result1 = corert.executeop(%gpu1) "tf.Relu"(%gpu_handle_input1) : 1

  %cpu_handle_result1 = corert.executeop(%gpu1) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result1) : 1
  %ch_print_gpu1 = corert.executeop.seq(%gpu1, %ch0) "tfrt_test.print"(%gpu_handle_result1) : 0
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 5], values = [0, 0, 0, 0.5, 1]
  %ch_print_cpu1 = corert.executeop.seq(%gpu1, %ch_print_cpu0) "tfrt_test.print"(%cpu_handle_result1) : 0

  tfrt.return %ch_print_cpu1 : !tfrt.chain
}

// CHECK: --- Running 'BM_Tf_Conv2d_1x256x56x56_3x3_256_f32'
func @BM_Tf_Conv2d_1x256x56x56_3x3_256_f32() {
  %ch_epoch = tfrt.new.chain
  %gpu0 = corert.get_op_handler %ch_epoch "gpu0"
  %gpu1 = corert.get_op_handler %ch_epoch "gpu1"

  %input0 = corert.executeop(%gpu0) "tfrt_test.create_dense_tensor"()
    { shape = [1, 256, 56, 56], values = [1.0 : f32] } : 1

  %filter0 = corert.executeop(%gpu0) "tfrt_test.create_dense_tensor"()
    { shape = [3, 3, 256, 256], values = [1.0 : f32] } : 1

  %input1 = corert.executeop(%gpu1) "tfrt_test.create_dense_tensor"()
    { shape = [1, 256, 56, 56], values = [1.0 : f32] } : 1

  %filter1 = corert.executeop(%gpu1) "tfrt_test.create_dense_tensor"()
    { shape = [3, 3, 256, 256], values = [1.0 : f32] } : 1


  tfrt_test.benchmark "BM_Tf_Conv2d_1x256x56x56_3x3_256_f32"(
      %gpu0      : !corert.device,
      %input0    : !corert.tensorhandle,
      %filter0   : !corert.tensorhandle,
      %gpu1      : !corert.device,
      %input1    : !corert.tensorhandle,
      %filter1   : !corert.tensorhandle
  )
  duration_secs = 2, max_count = 1000, num_warmup_runs = 10
  {
    %result0 = corert.executeop(%gpu0)
      "tf.Conv2D"(%input0, %filter0)
      {
        T = f32,
        data_format= "NCHW",
        padding = "VALID",
        strides = [1, 1, 1, 1],
        dilations = [1, 1, 1, 1]
      } : 1

    %done0 = corert.executeop(%gpu0) "tfrt_test.synchronize"(%result0) : 1

    %result1 = corert.executeop(%gpu1)
      "tf.Conv2D"(%input1, %filter1)
      {
        T = f32,
        data_format= "NCHW",
        padding = "VALID",
        strides = [1, 1, 1, 1],
        dilations = [1, 1, 1, 1]
      } : 1

    %done1 = corert.executeop(%gpu1) "tfrt_test.synchronize"(%result1) : 1


    tfrt.return %done1 : !corert.tensorhandle
  }

  tfrt.return
}
