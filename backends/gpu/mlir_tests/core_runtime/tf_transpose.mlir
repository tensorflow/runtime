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

// RUN: bef_executor -devices=cpu,gpu $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running 'transpose_1x0'
func @transpose_1x0() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %transpose_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = i64, value = dense<[[1,2,3],[4,5,6]]> : tensor<2x3xi64> } : 1

  %transpose_th = corert.executeop(%gpu) "_tf.Transpose"(%transpose_in_th1)
    { perm = dense<[1, 0]> : tensor<2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%transpose_th) : 1

  // CHECK: DenseHostTensor dtype = I64, shape = [3, 2], values = [1, 4, 2, 5, 3, 6]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'transpose_0x3x1x2'
func @transpose_0x3x1x2() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %transpose_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = i64, value = dense<[[[[ 1, 2, 3, 4],
                                      [ 5, 6, 7, 8]],
                                     [[ 9,10,11,12],
                                      [13,14,15,16]]]]> : tensor<1x2x2x4xi64> } : 1

  %transpose_th = corert.executeop(%gpu) "_tf.Transpose"(%transpose_in_th1)
    { perm = dense<[0, 3, 1, 2]> : tensor<4xi64> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%transpose_th) : 1

  // CHECK: DenseHostTensor dtype = I64, shape = [1, 4, 2, 2], values = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'transpose_1x0_f16'
func @transpose_1x0_f16() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %transpose_in_th1_f32 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32,  4.0 : f32, 5.0 : f32, 6.0 : f32] } : 1
  %transpose_in_th1_f16 = corert.executeop(%gpu)
    "tf.Cast"(%transpose_in_th1_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1

  %transpose_th_f16 = corert.executeop(%gpu) "_tf.Transpose"(%transpose_in_th1_f16)
    { perm = dense<[1, 0]> : tensor<2xi64> } : 1
  %transpose_th_f32 = corert.executeop(%gpu)
    "tf.Cast"(%transpose_th_f16) {DstT = f32, SrcT = f16, Truncate = true} : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%transpose_th_f32) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [3, 2], values = [1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
