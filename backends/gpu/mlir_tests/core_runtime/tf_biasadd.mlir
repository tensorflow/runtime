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

// RUN: bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'bias_add_f32'
func @bias_add_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [3], values = [-5.0 : f32, -4.0 : f32, -3.0 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.BiasAdd"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3], values = [-4, -2, 0, -1, 1, 3]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

func @bias_add_with_attrs() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %bias_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } : 1
  %bias_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<2xf32> } : 1

  // Note that data_format = "NHWC" is ignored since the input tensor is 2D.
  %bias_add_th = corert.executeop(%gpu) "tf.BiasAdd"(%bias_th1, %bias_th2)
    { data_format = "NHWC"} : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%bias_add_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [2, 2, 2, 2]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
