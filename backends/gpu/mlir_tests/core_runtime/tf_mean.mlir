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

// TODO(ezhulenev): Replace with: bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=fail.
// RUN: true

// CHECK: --- Running 'mean'
func @mean() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %mean_in_th1 = corert.executeop(%gpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<1x1x4x4xf32> } : 1
  %mean_in_th2 = corert.executeop(%gpu) "tf.Const"()
    { dtype = i32, value = dense<[2, 3]> : tensor<2xi32> } : 1
  %mean_th = corert.executeop(%gpu) "tf.Mean"(%mean_in_th1, %mean_in_th2) { T = f32, Tidx = i32, keep_dims = false } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%mean_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [1.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'mean_folded'
func @mean_folded() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %mean_in_th1 = corert.executeop(%gpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<1x1x4x4xf32> } : 1
  %mean_th = corert.executeop(%gpu) "_tf.Mean"(%mean_in_th1)
    { T = f32, Tidx = i32, keep_dims = false, reduction_indices = dense<[2, 3]> : tensor<2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "test.gpu_tensor_to_host_tensor"(%mean_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [1.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
