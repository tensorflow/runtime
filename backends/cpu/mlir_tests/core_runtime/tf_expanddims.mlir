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

// RUN: bef_executor -devices=cpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'expand_dims_i32'
func @expand_dims_i32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1

  %axis_zero = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [0 : i32] } : 1
  %axis_one = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [1 : i32] } : 1
  %axis_neg_one = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [-1 : i32] } : 1

  %cpu_handle_result_0 = corert.executeop(%cpu) "tf.ExpandDims"(%operand_0, %axis_zero) : 1
  %cpu_handle_result_1 = corert.executeop(%cpu) "tf.ExpandDims"(%operand_0, %axis_one) : 1
  %cpu_handle_result_2 = corert.executeop(%cpu) "tf.ExpandDims"(%operand_0, %axis_neg_one) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 2, 3]
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 1, 3]
  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3, 1]
  %ch_print_cpu_0 = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result_0) : 0
  %ch_print_cpu_1 = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result_1) : 0
  %ch_print_cpu_2 = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result_2) : 0

  %ch_print_cpu = tfrt.merge.chains %ch_print_cpu_0, %ch_print_cpu_1, %ch_print_cpu_2

  tfrt.return %ch_print_cpu : !tfrt.chain
}
