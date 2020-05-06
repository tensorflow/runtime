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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices=cpu | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'test_tensor_policy'
func @test_tensor_policy() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu)
      "tfrt_test.create_dense_tensor"() { shape = [5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.odd_collector"(%a_handle) : 1

   // CHECK: DenseHostTensor dtype = I32, shape = [3], values = [1, 3, 5]
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch3 : !hex.chain
}

// CHECK-LABEL: --- Running 'create_from_scalar_error'
func @create_from_scalar_error() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  // expected-error @+1 {{test.create_from_scalar must have 'value' attribute}}
  %a_handle = "corert.executeop"(%cpu) {op_attrs = [["shape", [2 : i64, 3 : i64]]], op_name = "tfrt_test.create_from_scalar"}
    : (!corert.device) -> !corert.tensorhandle

  hex.return %ch0 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_scalar_tensor_ops'
func @test_scalar_tensor_ops() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_from_scalar"() { shape = [2, 3], value = 1 : i32 } : 1

  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 3], value = 1
  %ch2 = "corert.print_tensorhandle"(%a_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %b_handle = corert.executeop(%cpu) "tfrt_test.add"(%a_handle, %a_handle) : 1

  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 3], value = 2
  %ch4 = "corert.print_tensorhandle"(%b_handle, %ch2) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch4 : !hex.chain
}


// CHECK-LABEL: --- Running 'test_scalar_dense_mixed'
func @test_scalar_dense_mixed() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    { shape = [2, 3], value = 1 : i32 } : 1

  // CHECK: ScalarHostTensor dtype = I32, shape = [2, 3], value = 1
  %ch2 = "corert.print_tensorhandle"(%a_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 3], values = [2 : i32] } : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 3], values = [2, 2, 2, 2, 2, 2]
  %ch4 = "corert.print_tensorhandle"(%b_handle, %ch2) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %c_handle = corert.executeop(%cpu) "tfrt_test.add"(%a_handle, %b_handle) : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 3], values = [3, 3, 3, 3, 3, 3]
  %ch6 = "corert.print_tensorhandle"(%c_handle, %ch4) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch6 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_scalar_denseonly_mixed'
func @test_scalar_denseonly_mixed() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    {shape = [2, 3], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [2, 3], values = [2: i32]} : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.add.denseonly"(%a_handle, %b_handle) : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 3], values = [3, 3, 3, 3, 3, 3]
  %ch4 = "corert.print_tensorhandle"(%c_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch4 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_scalar_denseonly2_mixed'
func @test_scalar_denseonly2_mixed() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    {shape = [2, 3], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [2, 3], values = [2: i32]} : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.add.denseonly2"(%a_handle, %b_handle) : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 3], values = [3, 3, 3, 3, 3, 3]
  %ch4 = "corert.print_tensorhandle"(%c_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch4 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_scalar_denseonly3_mixed'
func @test_scalar_denseonly3_mixed() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    {shape = [2, 3], value = 1: i32} : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [2, 3], values = [2: i32]} : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.add.denseonly3"(%a_handle, %b_handle) : 1

  // CHECK: future TensorHandle with metadata I32 [2, 3]
  %ch1 = "corert.print_tensorhandle"(%c_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  //  DenseHostTensor dtype = I32, shape = [2, 3], values = [3, 3, 3, 3, 3, 3]
  %ch2 = corert.executeop.seq(%cpu, %ch1) "tfrt_test.print"(%c_handle) : 0

  hex.return %ch2 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_coo_tensor'
func @test_coo_tensor() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [2, 2], values = [0 : i64, 0 : i64, 1 : i64, 2 : i64] } : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
   {shape = [2], values = [1 : i32, 1 : i32] } : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.create_coo_tensor"
    (%a_handle, %b_handle) {shape = [2, 3]} : 1

  // CHECK: CooHostTensor dtype = I32, shape = [2, 3], indices = [0, 0, 1, 2], values = [1, 1]
  %ch1 = "corert.print_tensorhandle"(%c_handle, %ch0)
    : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %d_handle = corert.executeop(%cpu) "tfrt_test.add"(%c_handle, %c_handle) : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [2, 3], values = [2, 0, 0, 0, 0, 2]
  %ch2 = "corert.print_tensorhandle"(%d_handle, %ch1)
    : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch2 : !hex.chain
}


// CHECK-LABEL: --- Running 'test_coo_scalar_mixed'
func @test_coo_scalar_mixed() -> !hex.chain {
  %ch0 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %a_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [1, 2], values = [0 : i64, 0 : i64] } : 1

  %b_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
   {shape = [1], values = [1 : i32] } : 1

  %c_handle = corert.executeop(%cpu) "tfrt_test.create_coo_tensor"
    (%a_handle, %b_handle) {shape = [1, 1]} : 1

  // CHECK: CooHostTensor dtype = I32, shape = [1, 1], indices = [0, 0], values = [1]
  %ch1 = "corert.print_tensorhandle"(%c_handle, %ch0)
    : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %d_handle = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
    {shape = [1, 1], value = 1: i32} : 1

  %e_handle = corert.executeop(%cpu) "tfrt_test.add"(%c_handle, %d_handle) : 1

  // CHECK: ScalarHostTensor dtype = I32, shape = [1, 1], value = 2
  %ch2 = "corert.print_tensorhandle"(%e_handle, %ch1)
    : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch2 : !hex.chain
}



