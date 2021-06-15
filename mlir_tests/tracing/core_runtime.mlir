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

// RUN: bef_executor_debug_tracing --enable_tracing --tracing_level=debug --test_init_function=register_op_handlers_cpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: Scope:Function: register_op_handlers_cpu
func @register_op_handlers_cpu() {
  // CHECK: Scope:corert.create_null_op_handler
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler

  // CHECK: Scope:corert.create_cpu_op_handler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler

  // CHECK: Scope:corert.register_op_handler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: Scope:Function: basic_test_matmul_f32
func @basic_test_matmul_f32() -> !tfrt.chain {

  // CHECK: Scope:tfrt.new.chain
  %ch0 = tfrt.new.chain

  // CHECK: Scope:corert.get_op_handler
  %cpu = corert.get_op_handler %ch0 "cpu"

  // CHECK: Scope:corert.executeop
  // CHECK: Scope:tfrt_test.create_dense_tensor#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatchFunction: tfrt_test.create_dense_tensor#op_name=tfrt_test.create_dense_tensor,Inputs=(),Results=(f32 [1, 65536]),Attributes=OpAttrs contains 2 entries:{{[[:space:]]*}}'shape' type=I64 value=[1, 65536]{{[[:space:]]*}}'values' type=F32 value=[1.000000e+00]{{[[:space:]]*}}#
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 65536], values = [1.0 : f32] } : 1

  // CHECK: Scope:corert.executeop
  // CHECK: Scope:tfrt_test.create_dense_tensor#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatchFunction: tfrt_test.create_dense_tensor#op_name=tfrt_test.create_dense_tensor,Inputs=(),Results=(f32 [65536, 1]),Attributes=OpAttrs contains 2 entries:{{[[:space:]]*}}'shape' type=I64 value=[65536, 1]{{[[:space:]]*}}'values' type=F32 value=[1.000000e+00]{{[[:space:]]*}}#
  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [65536, 1], values = [1.0 : f32] } : 1

  // CHECK: Scope:corert.executeop
  // CHECK: Scope:tfrt_test.matmul#op_handler=cpu#
  // CHECK: Scope:RunMetadataFunction
  // CHECK: Scope:RunDispatchFunction: tfrt_test.matmul#op_name=tfrt_test.matmul,long_name=MyCustomNameScope/MyCustomName,Inputs=(f32 [1, 65536];f32 [65536, 1]),Results=(f32 [1, 1]),Attributes=OpAttrs contains 2 entries:{{[[:space:]]*}}'transpose_a' type=BOOL value=0{{[[:space:]]*}}'transpose_b' type=BOOL value=0{{[[:space:]]*}}#
  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %b_handle)
    {transpose_a = false, transpose_b = false}: 1 loc("MyCustomNameScope/MyCustomName")

  tfrt.return %ch0 : !tfrt.chain
}
