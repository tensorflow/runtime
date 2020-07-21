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

// RUN: env CUDNN_LOGINFO_DBG=1 TFRT_DEBUG_DEFAULT_CONV_FWD_ALGO=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM bef_executor -devices=cpu,gpu $(bef_name %s) | FileCheck %s --dump-input=fail
// CHECK: --- Running 'conv2d_f32'
func @conv2d_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_input = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1, 2, 2], values = [-2.0 : f32, -1.0 : f32, 1.0 : f32,  2.0 : f32] } : 1

  %gpu_handle_filter = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [3, 3, 1, 1], values = [3.0 : f32, 0.0 : f32, 5.0 : f32,0.0 : f32, 0.0 : f32, 0.0 : f32,7.0 : f32, 0.0 : f32, 9.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.Conv2D"(%gpu_handle_input, %gpu_handle_filter) { data_format = "NCHW", padding = "SAME" } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [18, 7, -5, -6]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_f16'
func @conv2d_f16() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %conv2d_in_th1_f32 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2_f32 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1

  %conv2d_in_th1_f16 = corert.executeop(%gpu)
    "tf.Cast"(%conv2d_in_th1_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1
  %conv2d_in_th2_f16 = corert.executeop(%gpu)
    "tf.Cast"(%conv2d_in_th2_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1

  %conv2d_th_f16 = corert.executeop(%gpu) "tf.Conv2D"(%conv2d_in_th1_f16, %conv2d_in_th2_f16)
      {T = f32, data_format = "NCHW",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 2], use_cudnn_on_gpu = true}  : 1

  %conv2d_th_f32 = corert.executeop(%gpu)
    "tf.Cast"(%conv2d_th_f16) {DstT = f32, SrcT = f16, Truncate = true} : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%conv2d_th_f32) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 4, 1, 1],
  // CHECK-SAME: values = [3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_valid'
func @conv2d_valid() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %conv2d_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%gpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NCHW",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%conv2d_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 4, 2, 2], values = [{{(36, ){32}... }}]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_valid_strides'
func @conv2d_valid_strides() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %conv2d_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%gpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NCHW",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 2], use_cudnn_on_gpu = true}  : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%conv2d_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 4, 1, 1], values = [{{(36, ){15}36}}]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_same'
func @conv2d_same() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %conv2d_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%gpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NCHW",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%conv2d_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 4, 4, 4], values = [{{(16, 24, 24, 16, 24, 36, 36, 24, 24, 36, 36, 24, 16, 24, 24, 16, ){2}... }}]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_same_strides'
func @conv2d_same_strides() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %conv2d_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%gpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NCHW",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 2], use_cudnn_on_gpu = true}  : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%conv2d_th) : 1
  // Before changing to match TF behavior, answer is (16, 24, 24, 36, ){8}.
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 4, 2, 2], values = [{{(36, 24, 24, 16, ){8}... }}]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
