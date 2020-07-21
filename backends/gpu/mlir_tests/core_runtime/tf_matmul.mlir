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

// CHECK: --- Running 'matmul_2x2_by_2x2_f32'
func @matmul_2x2_by_2x2_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [0.5 : f32, 0.25 : f32, 0.125 : f32, 0.0625 : f32] } : 1
  %b = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 4.0 : f32, 8.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.MatMul"(%a, %b)
      { transpose_a = false, transpose_b = false} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [1.5, 3, 0.375, 0.75]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'matmul_2x2_trans_by_2x2_f32'
func @matmul_2x2_trans_by_2x2_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [0.5 : f32, 0.25 : f32, 0.125 : f32, 0.0625 : f32] } : 1
  %b = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 4.0 : f32, 8.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.MatMul"(%a, %b)
      { transpose_a = true, transpose_b = false} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [1, 2, 0.5, 1]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'matmul_2x2_by_2x2_trans_f32'
func @matmul_2x2_by_2x2_trans_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [0.5 : f32, 0.25 : f32, 0.125 : f32, 0.0625 : f32] } : 1
  %b = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 4.0 : f32, 8.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.MatMul"(%a, %b)
      { transpose_a = false, transpose_b = true} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [1, 4, 0.25, 1]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'matmul_2x2_trans_by_2x2_trans_f32'
func @matmul_2x2_trans_by_2x2_trans_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [0.5 : f32, 0.25 : f32, 0.125 : f32, 0.0625 : f32] } : 1
  %b = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 4.0 : f32, 8.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.MatMul"(%a, %b)
      { transpose_a = true, transpose_b = true} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [0.75, 3, 0.375, 1.5]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'matmul_2x2_by_2x2_f64'
func @matmul_2x2_by_2x2_f64() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [4.656612873077393e-10 : f64, 2.3283064365386963e-10 : f64, 1.1641532182693481e-10 : f64, 5.820766091346741e-11 : f64] } : 1
  %b = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f64, 2.0 : f64, 4.0 : f64, 8.0 : f64] } : 1

  %gpu_handle_result = corert.executeop(%gpu)
    "tf.MatMul"(%a, %b)
      { transpose_a = false, transpose_b = false} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F64, shape = [2, 2], values = [1.39698386192321{{.*}}e-09, 2.79396772384643{{.*}}e-09, 3.49245965480804{{.*}}e-10, 6.9849193096160{{.*}}e-10]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'matmul_2x2_by_2x2_f16'
func @matmul_2x2_by_2x2_f16() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %a_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [0.5 : f32, 0.25 : f32, 0.125 : f32, 0.0625 : f32] } : 1
  %b_f32 = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
      { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 4.0 : f32, 8.0 : f32] } : 1
  %a_f16 = corert.executeop(%gpu)
    "tf.Cast"(%a_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1
  %b_f16 = corert.executeop(%gpu)
    "tf.Cast"(%b_f32) {DstT = f16, SrcT = f32, Truncate = true} : 1

  %gpu_handle_result_f16 = corert.executeop(%gpu)
    "tf.MatMul"(%a_f16, %b_f16)
      { transpose_a = false, transpose_b = false} : 1

  %gpu_handle_result_f32 = corert.executeop(%gpu)
    "tf.Cast"(%gpu_handle_result_f16) {DstT = f32, SrcT = f16, Truncate = true} : 1

  %cpu_handle_result = corert.executeop(%gpu)
    "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result_f32) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [1.500000e+00, 3.000000e+00, 3.750000e-01, 7.500000e-01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
