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

// RUN: env CUDNN_LOGINFO_DBG=1 bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'fused_batch_norm_v3'
func @fused_batch_norm_v3() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  // Test tf.FusedBatchNormV3.
  %input = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<[[[[1.0, -1.0],[-1.0, 1.0]]]]> : tensor<1x1x2x2xf32> } : 1
  %scale = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1
  %bias = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %mean = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %variance = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1
  %res: 6 = corert.executeop(%gpu) "tf.FusedBatchNormV3"(%input, %scale, %bias, %mean, %variance)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NCHW", is_training = false } : 6

  %cpu_res = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%res#0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [1, -1, -1, 1]
  %ch_print = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_res) : 0
  tfrt.return %ch_print : !tfrt.chain
}

// CHECK: --- Running '_FusedBatchNormEx'
func @_FusedBatchNormEx() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  // Test _FusedBatchNormEx.
  %input = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<[[[[1.0, -1.0],[-1.0, 1.0]]]]> : tensor<1x1x2x2xf32> } : 1
  %scale = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1
  %bias = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %mean = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %variance = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1

  // With no side input.
  %res_no_side_input_identity: 6 = corert.executeop(%gpu) "tf._FusedBatchNormEx"(%input, %scale, %bias, %mean, %variance)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NCHW", activation_mode = "Identity" } : 6

  %cpu_res_no_side_input_identity = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%res_no_side_input_identity#0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [1, -1, -1, 1]
  %ch_print_0 = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_res_no_side_input_identity) : 0

  %res_no_side_input_relu: 6 = corert.executeop(%gpu) "tf._FusedBatchNormEx"(%input, %scale, %bias, %mean, %variance)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NCHW", activation_mode = "Relu" } : 6
  %cpu_res_no_side_input_relu = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%res_no_side_input_relu#0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [1, 0, 0, 1]
  %ch_print_1 = corert.executeop.seq(%gpu, %ch_print_0) "tfrt_test.print"(%cpu_res_no_side_input_relu) : 0

  // With side input.
  %side_input = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1

  %res_add_side_input_identity: 6 = corert.executeop(%gpu) "tf._FusedBatchNormEx"(%input, %scale, %bias, %mean, %variance, %side_input)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NCHW", activation_mode = "Identity" } : 6
  %cpu_res_add_side_input_identity = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%res_add_side_input_identity#0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [2, 0, 0, 2]
  %ch_print_2 = corert.executeop.seq(%gpu, %ch_print_1) "tfrt_test.print"(%cpu_res_add_side_input_identity) : 0

  // Note that res = Relu(BN(x) + side_input)
  %res_add_side_input_relu: 6 = corert.executeop(%gpu) "tf._FusedBatchNormEx"(%input, %scale, %bias, %mean, %variance, %side_input)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NCHW", activation_mode = "Relu" } : 6
  %cpu_res_add_side_input_relu = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%res_add_side_input_relu#0) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [2, 0, 0, 2]
  %ch_print_3 = corert.executeop.seq(%gpu, %ch_print_2) "tfrt_test.print"(%cpu_res_add_side_input_relu) : 0

  tfrt.return %ch_print_3 : !tfrt.chain
}

