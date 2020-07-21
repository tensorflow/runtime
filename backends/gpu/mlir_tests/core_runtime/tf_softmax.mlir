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

// CHECK: --- Running 'softmax_f32'
func @softmax_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_input = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [3, 3], values = [-2.0 : f32, -1.5 : f32, -1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32, 2.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu) "tf.Softmax"(%gpu_handle_input) : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [3, 3], values = [0.0072{{.*}}, 0.0120{{.*}}, 0.0198{{.*}}, 0.0326{{.*}}, 0.0538{{.*}}, 0.0887{{.*}}, 0.1463{{.*}}, 0.2413{{.*}}, 0.3978{{.*}}]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'softmax'
func @softmax() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %softmax_in_th = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %softmax_th = corert.executeop(%gpu) "tf.Softmax"(%softmax_in_th) { T = f32 } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%softmax_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 1, 1], values = [1]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
