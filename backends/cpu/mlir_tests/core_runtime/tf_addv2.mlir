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

// CHECK: --- Running 'addV2_dense_dense_f32'
func @addV2_dense_dense_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [0.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [-1.000000e+00, 5.000000e-01, 2.000000e+00, 3.500000e+00, 5.000000e+00, 6.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_dense_scalar_f32'
func @addV2_dense_dense_scalar_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [1.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_scalar_dense_f32'
func @addV2_dense_scalar_dense_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1], values = [1.0 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_dense_bcast_f32'
func @addV2_dense_dense_bcast_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [0.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 4.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_bcast_dense_f32'
func @addV2_dense_bcast_dense_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [0.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 4.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_bcast_dense_bcast_f32'
func @addV2_dense_bcast_dense_bcast_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 1], values = [4.0 : f32, 5.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [5.000000e+00, 6.000000e+00, 7.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_dense_scalar_f32'
func @addV2_dense_scalar_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    { shape = [2, 3], value = 1.0: f32 } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_scalar_dense_f32'
func @addV2_scalar_dense_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    { shape = [2, 3], value = 1.0: f32 } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [0.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'addV2_scalar_scalar_f32'
func @addV2_scalar_scalar_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    { shape = [2, 3], value = 1.0: f32 } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    { shape = [2, 3], value = 2.0: f32 } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.AddV2"(%operand_0, %operand_1) : 1
  // CHECK: ScalarHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: value = 3.000000e+00
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
