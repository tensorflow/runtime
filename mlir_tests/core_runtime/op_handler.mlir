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

// CHECK-LABEL: --- Not running 'register_op_handlers' because it has arguments.
func @register_op_handlers(%ch0: !tfrt.chain) -> !tfrt.chain {
  %null = corert.get_op_handler %ch0 "null"
  %ch1 = corert.register_op_handler %null "custom0"
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'get_op_handler' because it has arguments.
func @get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  %null = corert.get_op_handler %ch0 "custom0"
  %ch1 = tfrt.new.chain
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'failed_get_op_handler' because it has arguments.
func @failed_get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  // expected-error @+1 {{runtime error: op_handler not found}}
  %null = corert.get_op_handler %ch0 "custom0"
  %ch1 = tfrt.new.chain
  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_op_handler_chain_registration'
func @test_op_handler_chain_registration()  -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @failed_get_op_handler(%ch0) : (!tfrt.chain) -> !tfrt.chain
  %ch2 = tfrt.call @register_op_handlers(%ch1) : (!tfrt.chain) -> !tfrt.chain
  %ch3 = tfrt.call @get_op_handler(%ch2) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return %ch3 : !tfrt.chain
}
