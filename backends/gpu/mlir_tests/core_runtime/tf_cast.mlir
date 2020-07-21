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

// CHECK: --- Running 'cast_f64_to_f32_no_truncate'
func @cast_f64_to_f32_no_truncate() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  // the large numbers are {+,-}0x1.ffffffp+127, which just exceeds f32 limit ({+,-}0x1.fffffep+127).
  // Truncation makes it non-inf, while no truncation results in inf.
  %input = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [-340282356779733661637539395458142568448.000000 : f64, -0.5 : f64, 0.0 : f64, 0.5 : f64, 1.0 : f64, 340282356779733661637539395458142568448.000000 : f64] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.Cast"(%input) {DstT = f32, SrcT = f64, Truncate = false} : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3], values = [-inf, -0.5, 0, 0.5, 1, inf]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'cast_f64_to_f32_with_truncate'
func @cast_f64_to_f32_with_truncate() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %input = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [-340282356779733661637539395458142568448.000000 : f64, -0.5 : f64, 0.0 : f64, 0.5 : f64, 1.0 : f64, 340282356779733661637539395458142568448.000000 : f64] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.Cast"(%input) {DstT = f32, SrcT = f64, Truncate = true} : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3], values = [-3.40282347e+38, -0.5, 0, 0.5, 1, 3.40282347e+38]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'cast_f32_to_f64_no_truncate'
func @cast_f32_to_f64_no_truncate() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %input = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [-3.4028234663852886e+38 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 3.4028234663852886e+38 : f32] } : 1
  %gpu_handle_result = corert.executeop(%gpu) "tf.Cast"(%input) {DstT = f64, SrcT = f64, Truncate = false} : 1
  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F64, shape = [2, 3], values = [-3.4028234663852886e+38, -0.5, 0, 0.5, 1, 3.4028234663852886e+38]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
