// Copyright 2021 The TensorFlow Runtime Authors
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

// RUN: bef_executor_debug_tracing --enable_tracing --tracing_level=debug --test_init_function=register_op_handlers_cpu %s.bef | FileCheck %s

// CHECK: Scope:Function: register_op_handlers_cpu
func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: Scope:Function: basic_test_matmul_f32
func @basic_test_matmul_f32() -> !tfrt.chain {

  %ch0 = tfrt.new.chain

  %cpu = corert.get_op_handler %ch0 "cpu"

  // CHECK: Scope:tfrt_test.create_dense_tensor#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatch: tfrt_test.create_dense_tensor
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 65536], values = [1.0 : f32] } : 1

  // CHECK: Scope:tfrt_test.create_dense_tensor#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatch: tfrt_test.create_dense_tensor
  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [65536, 1], values = [1.0 : f32] } : 1

  // CHECK: Scope:tfrt_test.matmul#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatch: tfrt_test.matmul
  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1 loc("MyCustomNameScope/MyCustomName")

  tfrt.return %ch0 : !tfrt.chain
}
