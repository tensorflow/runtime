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

// CHECK: --- Running 'addv2'
func @addv2() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %addv2_in_th1 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1
  %addv2_in_th2 = corert.executeop(%gpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<2xf32> } : 1
  %add_th = corert.executeop(%gpu) "tf.AddV2"(%addv2_in_th1, %addv2_in_th2) { T = f32 } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%add_th) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [2, 2, 2, 2]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_f16'
func @addV2_f16() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [1.2 : f32, 1.2 : f32, 1.2 : f32, 1.2 : f32, 1.2 : f32] } : 1
  %operand_0_f16 = corert.executeop(%gpu)
    "tf.Cast"(%operand_0) {DstT = f16, SrcT = f32, Truncate = true} : 1
  %operand_1_f16 = corert.executeop(%gpu)
    "tf.Cast"(%operand_1) {DstT = f16, SrcT = f32, Truncate = true} : 1

  %gpu_handle_result_f16 = corert.executeop(%gpu) "tf.AddV2"(%operand_0_f16, %operand_1_f16) : 1

  %gpu_handle_result_fp32 = corert.executeop(%gpu)
    "tf.Cast"(%gpu_handle_result_f16) {DstT = f32, SrcT = f16, Truncate = true} : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result_fp32) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [5], values = [1.992188e-01, 6.992188e-01, 1.199219e+00, 1.699219e+00, 2.199219e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_f32'
func @addV2_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [1.2 : f32, 1.2 : f32, 1.2 : f32, 1.2 : f32, 1.2 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [5], values = [0.200000048, 0.700000048, 1.20000005, 1.70000005, 2.20000005]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_broadcast0_f32'
func @addV2_broadcast0_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [1.2 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [5], values = [0.200000048, 0.700000048, 1.20000005, 1.70000005, 2.20000005]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_broadcast1_f32'
func @addV2_broadcast1_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 2.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [1.2 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3], values = [0.200000048, 0.700000048, 1.20000005, 1.70000005, 2.20000005, 3.20000005]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}


// CHECK: --- Running 'addV2_broadcast2_f32'
func @addV2_broadcast2_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 2], values = [1.0 : f32, 2.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 1], values = [5.0 : f32, 6.0 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [6, 7, 7, 8]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_broadcast3_f32'
func @addV2_broadcast3_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 2.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [3], values = [1.2 : f32, 2.2 : f32, 3.2 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3], values = [0.200000048, 1.70000005, 3.20000005, 1.70000005, 3.20000005, 5.19999981]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_broadcast4_f32'
func @addV2_broadcast4_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [8, 1, 6, 1], values = [0.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [7, 1, 5], values = [0.0 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [8, 7, 6, 5]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}


// CHECK: --- Running 'addV2_broadcast5_f32'
func @addV2_broadcast5_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2, 2, 2, 2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32, 7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32, 17.0 : f32, 18.0 : f32, 19.0 : f32, 20.0 : f32, 21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32, 27.0 : f32, 28.0 : f32, 29.0 : f32, 30.0 : f32, 31.0 : f32, 32.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32, 7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32, 17.0 : f32, 18.0 : f32, 19.0 : f32, 20.0 : f32, 21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32, 27.0 : f32, 28.0 : f32, 29.0 : f32, 30.0 : f32, 31.0 : f32, 32.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1, 1, 1, 1, 1, 1], values = [3.0 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 2, 2, 2, 2, 2, 2], values = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, ... ]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_broadcast6_f32'
func @addV2_broadcast6_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %operand_0 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1, 1, 1, 1, 1, 1], values = [3.0 : f32] } : 1
  %operand_1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2, 2, 2, 2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32, 7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32, 17.0 : f32, 18.0 : f32, 19.0 : f32, 20.0 : f32, 21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32, 27.0 : f32, 28.0 : f32, 29.0 : f32, 30.0 : f32, 31.0 : f32, 32.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32, 7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32, 17.0 : f32, 18.0 : f32, 19.0 : f32, 20.0 : f32, 21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32, 27.0 : f32, 28.0 : f32, 29.0 : f32, 30.0 : f32, 31.0 : f32, 32.0 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 2, 2, 2, 2, 2, 2], values = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, ... ]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
