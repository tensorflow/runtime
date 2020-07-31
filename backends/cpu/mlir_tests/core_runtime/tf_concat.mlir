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

// CHECK: --- Running 'concat_f32_axis_1'
func @concat_f32_axis_1() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32] } : 1

  %axis = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [1 : i32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.ConcatV2"(%operand_0, %operand_1, %axis)
    { N = 2 : i64 }: 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 6]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00,
  // CHECK-SAME:           4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'concat_f32_axis_neg_1'
func @concat_f32_axis_neg_1() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32] } : 1

  %axis = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [-1 : i32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.ConcatV2"(%operand_0, %operand_1, %axis)
    { N = 2 : i64 }: 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 6]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00,
  // CHECK-SAME:           4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'concat_f32_scalars'
func @concat_f32_scalars() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %s0 = corert.create_dense_tensor.f32 {shape = [], value = [0.125 : f32]}
  %s1 = corert.create_dense_tensor.f32 {shape = [], value = [0.250 : f32]}

  %axis = corert.create_dense_tensor.i32 {shape = [], value = [0 : i32]}

  %cpu_handle_result = corert.executeop(%cpu) "tf.ConcatV2"(%s0, %s1, %axis) { N = 2 : i64 }: 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2]
  // CHECK-SAME: values = [1.250000e-01, 2.500000e-01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
