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

// CHECK: --- Running 'forward_unary_op_argument'
func @forward_unary_op_argument() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1

  // CHECK: DenseHostTensor: buffer=[[ADDR:.*]]
  %ch1    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand) : 0
  %result = corert.executeop(%cpu) "tf.Log"(%operand) : 1
  // CHECK: DenseHostTensor: buffer=[[ADDR]]
  %ch2    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%result) : 0

  tfrt.return
}

// CHECK: --- Running 'do_not_forward_unary_op_argument'
func @do_not_forward_unary_op_argument() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1

  // CHECK: DenseHostTensor: buffer=[[ADDR:.*]]
  %ch1     = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand) : 0
  %result0 = corert.executeop(%cpu) "tf.Log"(%operand) : 1
  %result1 = corert.executeop(%cpu) "tf.Log"(%operand) : 1
  // CHECK-NOT: DenseHostTensor: buffer=[[ADDR]]
  %ch2     = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%result0) : 0

  tfrt.return
}

// CHECK: --- Running 'forward_binary_op_lhs_argument'
func @forward_binary_op_lhs_argument() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1
  %operand1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1

  // CHECK: DenseHostTensor: buffer=[[ADDR:.*]]
  %ch1    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand0) : 0
  %result = corert.executeop(%cpu) "tf.Mul"(%operand0, %operand1) : 1
  // CHECK: DenseHostTensor: buffer=[[ADDR]]
  %ch2    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%result) : 0

  tfrt.return
}

// CHECK: --- Running 'forward_binary_op_rhs_argument'
func @forward_binary_op_rhs_argument() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1
  %operand1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1

  // CHECK: DenseHostTensor: buffer=[[ADDR:.*]]
  %ch1    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand1) : 0
  %result = corert.executeop(%cpu) "tf.Mul"(%operand0, %operand1) : 1
  %log    = corert.executeop(%cpu) "tf.Log"(%operand0) : 1
  // CHECK: DenseHostTensor: buffer=[[ADDR]]
  %ch2    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%result) : 0

  tfrt.return
}

// CHECK: --- Running 'do_not_forward_binary_op_arguments'
func @do_not_forward_binary_op_arguments() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1
  %operand1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32] } : 1

  // CHECK: DenseHostTensor: buffer=[[ADDR0:.*]]
  // CHECK: DenseHostTensor: buffer=[[ADDR1:.*]]
  %ch1    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand0) : 0
  %ch2    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%operand1) : 0

  %result = corert.executeop(%cpu) "tf.Mul"(%operand0, %operand1) : 1
  %log0   = corert.executeop(%cpu) "tf.Log"(%operand0) : 1
  %log1   = corert.executeop(%cpu) "tf.Log"(%operand1) : 1

  // CHECK-NOT: DenseHostTensor: buffer=[[ADDR0]]
  // CHECK-NOT: DenseHostTensor: buffer=[[ADDR1]]
  %ch3    = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print_address"(%result) : 0

  tfrt.return
}
