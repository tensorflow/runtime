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

// RUN: bef_executor -devices=cpu,gpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'pad_i64'
func @pad_i64() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %pad_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = i64, value = dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi64> } : 1
  %pad_th = corert.executeop(%gpu) "_tf.Pad"(%pad_in_th1)
      { T = i64, Tpaddings = i32, paddings = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%pad_th) : 1
  // CHECK: DenseHostTensor dtype = I64, shape = [4, 7], values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'pad_f32'
func @pad_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %pad_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<[[1.,2.,3.],[4.,5.,6.]]> : tensor<2x3xf32> } : 1
  %pad_th = corert.executeop(%gpu) "_tf.Pad"(%pad_in_th1)
      { T = f32, Tpaddings = i32, paddings = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%pad_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 7], values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'pad_f16'
func @pad_f16() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %pad_in_th1_f32 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<[[1.,2.,3.],[4.,5.,6.]]> : tensor<2x3xf32> } : 1
  %pad_in_th1_f16 = corert.executeop(%gpu)
    "tf.Cast"(%pad_in_th1_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1

  // TODO(b/149063226): The T attribute here should be f16 instead of f32.
  // However, right bef does not support type f16. Fix it later.
  %pad_th_f16 = corert.executeop(%gpu) "_tf.Pad"(%pad_in_th1_f16)
      { T = f32, Tpaddings = i32, paddings = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32> } : 1

  %pad_th_f32 = corert.executeop(%gpu)
    "tf.Cast"(%pad_th_f16) {DstT = f32, SrcT = f16, Truncate = true} : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%pad_th_f32) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [4, 7], values = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'pad_with_hardcoded_attr_f32'
func @pad_with_hardcoded_attr_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [1, 1, 1, 1], values = [0.5 : f32] } : 1

  %padding = corert.executeop(%gpu) "tf.Const"()
      { dtype = i32, value = dense<[[0, 0], [0, 0], [3, 3], [3, 3]]> : tensor<4x2xi32> } : 1

  // %padding input is ignored.
  // Values are hardcoded.
  %gpu_handle_result = corert.executeop(%gpu)
    "tf.Pad"(%a, %padding) : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // TODO(tfrt-devs): The values may not be right.
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 7, 7], values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, ... ]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
