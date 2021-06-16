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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu $(bef_name %s) | FileCheck %s --dump-input=fail

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'concat_f32_axis_1'
func @concat_f32_axis_1() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : () -> !t.tensor
  %operand_1 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32] } : () -> !t.tensor

  %axis = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64, 6: i64]

  "tf_sync.ConcatV2.f32"(%axis, %operand_0, %operand_1, %result) : (i32, !t.tensor, !t.tensor, !t.tensor) -> ()

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 6]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00,
  // CHECK-SAME:           4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}

// CHECK: --- Running 'concat_f32_axis_neg_1'
func @concat_f32_axis_neg_1() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : () -> !t.tensor
  %operand_1 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32] } : () -> !t.tensor

  %axis = "tfrt.constant_s.i32"() {value = -1 : i32} : () -> i32

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64, 6: i64]

  "tf_sync.ConcatV2.f32"(%axis, %operand_0, %operand_1, %result) : (i32, !t.tensor, !t.tensor, !t.tensor) -> ()

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 6]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00,
  // CHECK-SAME:           4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}

// CHECK: --- Running 'concat_f32_scalars'
func @concat_f32_scalars() attributes {tfrt.sync} {
  %s0 = "tfrt_dht_sync.create_dense_tensor.f32"()
  {shape = [], value = [0.125 : f32]} : () -> !t.tensor
  %s1 = "tfrt_dht_sync.create_dense_tensor.f32"()
  {shape = [], value = [0.250 : f32]} : () -> !t.tensor

  %axis = "tfrt.constant_s.i32"() {value = 0 : i32} : () -> i32

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64]

  "tf_sync.ConcatV2.f32"(%axis, %s0, %s1, %result) : (i32, !t.tensor, !t.tensor, !t.tensor) -> ()

  // CHECK: DenseHostTensor dtype = f32, shape = [2]
  // CHECK-SAME: values = [1.250000e-01, 2.500000e-01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}

// CHECK: --- Running 'concat_zero_dim'
func @concat_zero_dim() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 0], values = [] } : () -> !t.tensor
  %operand_1 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [7.0 : f32, 8.0 : f32, 9.0 : f32, 10.0 : f32, 11.0 : f32, 12.0 : f32] } : () -> !t.tensor

  %axis = "tfrt.constant_s.i32"() {value = 1 : i32} : () -> i32

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64, 3: i64]

  "tf_sync.ConcatV2.f32"(%axis, %operand_0, %operand_1, %result) : (i32, !t.tensor, !t.tensor, !t.tensor) -> ()

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}
