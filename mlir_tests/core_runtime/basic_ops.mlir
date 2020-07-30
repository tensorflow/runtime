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

// CHECK-LABEL: --- Running 'basic_test_matmul_f32'
func @basic_test_matmul_f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Create tensor whose shape is represented using RepKind::kRep32.
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 65536], values = [1.0 : f32] } : 1

  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [65536, 1], values = [1.0 : f32] } : 1

  // Create tensor whose shape is represented using RepKind::kRep16.
  %c_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  // This test.matmul involves two tensors whose shapes are represented using
  // RepKind::kRep32.
  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [6.553600e+04]
  %ch5 = "corert.print_tensorhandle"(%result1, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // This test.matmul involves two tensors whose shapes are represented using
  // RepKind::kRep16.
  %result2 = corert.executeop(%cpu) "tfrt_test.matmul"(%result1, %c_handle)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [1.310720e+05]
  %ch7 = "corert.print_tensorhandle"(%result2, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch7 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic_test_matmul_transpose_f32'
func @basic_test_matmul_transpose_f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] } : 1

  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] } : 1

  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1

  %result2 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = true}: 1

  %result3 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = true, transpose_b = false}: 1

  %result4 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = true, transpose_b = true}: 1

  // CHECK: shape = [2, 2], values = [7.000000e+00, 1.000000e+01, 1.500000e+01, 2.200000e+01]
  %ch1 = "corert.print_tensorhandle"(%result1, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [5.000000e+00, 1.100000e+01, 1.100000e+01, 2.500000e+01]
  %ch2 = "corert.print_tensorhandle"(%result2, %ch1) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [1.000000e+01, 1.400000e+01, 1.400000e+01, 2.000000e+01]
  %ch3 = "corert.print_tensorhandle"(%result3, %ch2) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [2, 2], values = [7.000000e+00, 1.500000e+01, 1.000000e+01, 2.200000e+01]
  %ch4 = "corert.print_tensorhandle"(%result4, %ch3) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch4 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic_test_matmul_i32'
func @basic_test_matmul_i32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Create tensor whose shape is represented using RepKind::kRep32.
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 65536], values = [1 : i32] } : 1

  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [65536, 1], values = [1 : i32] } : 1

  // Create tensor whose shape is represented using RepKind::kRep16.
  %c_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2 : i32] } : 1

  // This test.matmul involves two tensors whose shapes are represented using
  // RepKind::kRep32.
  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [65536]
  %ch5 = "corert.print_tensorhandle"(%result1, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // This test.matmul involves two tensors whose shapes are represented using
  // RepKind::kRep16.
  %result2 = corert.executeop(%cpu) "tfrt_test.matmul"(%result1, %c_handle)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [131072]
  %ch7 = "corert.print_tensorhandle"(%result2, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch7 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'basic_test_ops'
func @basic_test_ops() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Create tensor whose shape is represented using RepKind::kRep32.
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 4], values = [1 : i32, 0 : i32, 2 : i32, 0 : i32] } : 1

  // add op.
  %result1 = corert.executeop(%cpu) "tfrt_test.add"(%a_handle, %a_handle) : 1

  // CHECK: shape = [1, 4], values = [2, 0, 4, 0]
  %ch3 = "corert.print_tensorhandle"(%result1, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // equal op.
  %result2 = corert.executeop(%cpu) "tfrt_test.equal"(%a_handle, %result1) : 1

  // CHECK: shape = [1, 4], values = [0, 1, 0, 1]
  %ch5 = "corert.print_tensorhandle"(%result2, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // argmax op.
  %result3 = corert.executeop(%cpu)
    "tfrt_test.argmax"(%a_handle) { axis = 1 : i32 } : 1

  // CHECK: shape = [1], values = [2]
  %ch7 = "corert.print_tensorhandle"(%result3, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // reduce_mean op.
  %result4 = corert.executeop(%cpu)
    "tfrt_test.reduce_mean"(%result3) { axis = 0 : i32 } : 1

  // CHECK: shape = [], values = [2]
  %ch9 = "corert.print_tensorhandle"(%result4, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch9 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'tensorhandle_to_shape_test'
func @tensorhandle_to_shape_test() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Create tensor whose shape is represented using RepKind::kRep32.
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 4], values = [1 : i32, 0 : i32, 2 : i32, 0 : i32] } : 1

  %a_shape = "corert.tensorhandle_to_shape"(%a_handle, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !ts.shape

  // CHECK: shape = [1, 4]
  "ts.print_shape"(%a_shape) : (!ts.shape) -> ()

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensorhandle_error_test'
func @tensorhandle_error_test() -> i32 {
  %ch0 = tfrt.new.chain
  %one = tfrt.constant.i32 1
  %cpu = corert.get_op_handler %ch0 "cpu"

  %tensor = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.i32 %tensor, %ch0 1 : i32

  // expected-error @+1 {{invalid tensor metadata}}
  %handle = "tfrt_test.tensorhandle_with_error_metadata"(%tensor, %ch1)
    : (!t.tensor, !tfrt.chain) -> !corert.tensorhandle

  %shape = "corert.tensorhandle_to_shape"(%handle, %ch1)
    : (!corert.tensorhandle, !tfrt.chain) -> !ts.shape

  // This line should not be executed because its input %shape has error.
  // It is validated by the CHECK-NEXT below.
  %ch2 = "tfrt_dht.print_tensor_shape"(%shape, %ch1) : (!ts.shape, !tfrt.chain) -> !tfrt.chain

  // CHECK-NEXT: 'tensorhandle_error_test' returned 1
  tfrt.return %one : i32
}

// CHECK-LABEL: --- Running 'badop_error'
func @badop_error() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // expected-error @+1 {{tf.invalidop was not supported by NullOpHandler}}
  %op_ch = corert.executeop.seq(%cpu, %ch0) "tf.invalidop"()

  tfrt.return
}

// CHECK-LABEL: --- Running 'shape_error'
func @shape_error() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 1], values = [2.0 : f32] } : 1

  // expected-error @+1 {{matmul arguments have incompatible shapes}}
  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1

  tfrt.return
}

// CHECK-LABEL: --- Running 'basic_executeop'
func @basic_executeop() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 3], values = [1 : i32] } : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [1, 3], values = [1, 1, 1]
  %ch3 = "corert.print_tensorhandle"(%a_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %ch4, %b_handle = corert.executeop.seq(%cpu, %ch3)
    "tfrt_test.create_dense_tensor"() { shape = [1, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32] } : 1

  // CHECK: shape = [1, 3], values = [1.000000e+00, 2.000000e+00, 3.000000e+00]
  %ch5 = "corert.print_tensorhandle"(%b_handle, %ch4) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch5 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_async'
func @test_async() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [2: i64, 2: i64], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.async.noop"(%a_handle) : 1

  // CHECK: future TensorHandle with metadata I32 [2, 2]
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 2], value = 1
  %op_ch4 = corert.executeop.seq(%cpu, %ch3) "tfrt_test.print"(%b_handle) : 0

  tfrt.return %op_ch4 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_async_no_md'
func @test_async_no_md() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [2: i64, 2: i64], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.async.noop_no_md"(%a_handle) : 1

  // CHECK: fully future TensorHandle with unresolved metadata
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_cancel'
func @test_cancel() -> !t.tensor{
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [2: i64, 2: i64], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.async.noop"(%a_handle) : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.async.noop"(%b_handle) : 1

  %c_ht = "corert.tensorhandle_to_ht"(%c_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !t.tensor

  %x, %ch1 = "tfrt_test.cancel"(%ch0) : (!tfrt.chain) -> (i32, !tfrt.chain)

  tfrt.return %c_ht : !t.tensor
}
// CHECK-NEXT: returned <<error: Cancelled>>

// CHECK-LABEL: --- Running 'test_side_effect'
func @test_side_effect() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [2: i64, 2: i64], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.async.noop"(%a_handle) : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.add"(%b_handle, %b_handle) : 1

  // CHECK: future TensorHandle with metadata I32 [2, 2]
  %ch4 = "corert.print_tensorhandle"(%c_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // Print in the opposite order from resolving to make sure the prints get
  // sequenced correctly.

  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 2], value = 2
  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 2], value = 1

  %op_ch5 = corert.executeop.seq(%cpu, %ch4) "tfrt_test.print"(%c_handle) : 0

  %op_ch6 = corert.executeop.seq(%cpu, %op_ch5) "tfrt_test.print"(%b_handle) : 0

  tfrt.return %op_ch6 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_error_propagation'
func @test_error_propagation() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [1: i64, 1: i64], value = 1: i32} : 1

  // expected-error @+1 {{runtime error: error from test.error.tensor implementation}}
  %b_handle = corert.executeop(%cpu) "tfrt_test.error.tensor"(%a_handle) : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.add"(%b_handle, %b_handle) : 1

  // This op should not run, given that the input is an error.
  %op_ch5 = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print"(%c_handle) : 0

  // CHECK-NEXT: Error TensorHandle: 'error from test.error.tensor implementation'
  %ch4 = "corert.print_tensorhandle"(%c_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch4 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_async_dispatch_async_metadata_error'
func @test_async_dispatch_async_metadata_error() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [1.0 : f32] } : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.async.noop_no_md"(%a_handle) : 1

  // CHECK: fully future TensorHandle with unresolved metadata
  %ch1 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %res_handle = corert.executeop(%cpu) "tfrt_test.matmul" (%a_handle, %b_handle) {transpose_a = false, transpose_b = false} : 1

  tfrt.return %ch1 : !tfrt.chain
}

func @return_first(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %x : !tfrt.chain, !corert.tensorhandle
}

func @return_second(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %y : !tfrt.chain, !corert.tensorhandle
}

// CHECK-LABEL: --- Running 'control_flow_conditional'
func @control_flow_conditional() {
  %ch0 = tfrt.new.chain

  %cpu = corert.get_op_handler %ch0 "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [1 : i32] } : 1
  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [2 : i32] } : 1

  %true_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [1 : i32] } : 1

  %true_res:2 = corert.cond %true_handle @return_first @return_second (%ch0, %a_handle, %b_handle) : (!corert.tensorhandle)

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 2], values = [1, 1, 1, 1]
  %ch2 = "corert.print_tensorhandle"(%true_res#1, %true_res#0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %false_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [0 : i8] } : 1

  %false_handle_unresolved = corert.executeop(%cpu) "tfrt_test.async.noop_no_md"(%false_handle) : 1

  %false_res:2 = corert.cond %false_handle_unresolved @return_first @return_second (%ch2, %a_handle, %b_handle) : (!corert.tensorhandle)

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 2], values = [2, 2, 2, 2]
  %ch3 = "corert.print_tensorhandle"(%false_res#1, %false_res#0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_string_tensor'
func @test_string_tensor() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a = corert.const_string_tensor {shape = [2], value = ["string", "tensor"]}

  // CHECK: StringHostTensor shape = [2], values = ["string", "tensor"]
  %ch1 = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print"(%a) : 0

  tfrt.return %ch1 : !tfrt.chain
}

