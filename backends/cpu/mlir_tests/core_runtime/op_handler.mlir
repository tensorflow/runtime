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

// RUN: bef_executor -devices=null $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Not running 'register_cpu_op_handler_chain' because it has arguments.
func @register_cpu_op_handler_chain(%ch0: !tfrt.chain) -> !tfrt.chain {
  %null = "corert.create_null_op_handler"() : () -> !corert.device
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.device) -> !corert.device
  %ch = corert.register_op_handler %cpu "cpu0"
  tfrt.return %ch : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'get_cpu_op_handler' because it has arguments.
func @get_cpu_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  %cpu0 = corert.get_op_handler %ch0 "cpu0"
  %cpu_handle_result = corert.executeop(%cpu0)
    "tf.Const"() {value = dense<[42, 314]> : tensor<2xi32>, dtype = i32} : 1

  %ch1 = corert.executeop.seq(%cpu0, %ch0) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'failed_cpu_get_op_handler' because it has arguments.
func @failed_cpu_get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  // expected-error @+1 {{runtime error: op_handler not found}}
  %cpu0 = corert.get_op_handler %ch0 "cpu0"
  %ch1 = tfrt.new.chain
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_cpu_op_handler_chain_kernels'
func @test_cpu_op_handler_chain_kernels()  -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @failed_cpu_get_op_handler(%ch0) : (!tfrt.chain) -> !tfrt.chain
  %ch2 = tfrt.call @register_cpu_op_handler_chain(%ch1) : (!tfrt.chain) -> !tfrt.chain
  // CHECK: DenseHostTensor dtype = I32, shape = [2], values = [42, 314]
  %ch3 = tfrt.call @get_cpu_op_handler(%ch2) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return %ch3 : !tfrt.chain
}
