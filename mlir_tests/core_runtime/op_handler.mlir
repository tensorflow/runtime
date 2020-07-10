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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices=null | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Not running 'register_op_handler_chain' because it has arguments.
func @register_op_handler_chain(%ch0: !hex.chain) -> !hex.chain {
  %null = corert.get_op_handler %ch0 "null"
  %ch1 = corert.register_op_handler_chain %null "custom0"
  hex.return %ch1 : !hex.chain
}

// CHECK-LABEL: --- Not running 'get_op_handler' because it has arguments.
func @get_op_handler(%ch0: !hex.chain) -> !hex.chain {
  %null = corert.get_op_handler %ch0 "custom0"
  %ch1 = hex.new.chain
  hex.return %ch1 : !hex.chain
}

// CHECK-LABEL: --- Not running 'failed_get_op_handler' because it has arguments.
func @failed_get_op_handler(%ch0: !hex.chain) -> !hex.chain {
  // expected-error @+1 {{runtime error: op_handler not found}}
  %null = corert.get_op_handler %ch0 "custom0"
  %ch1 = hex.new.chain
  hex.return %ch1 : !hex.chain
}

// CHECK-LABEL: --- Running 'test_op_handler_chain_registration'
func @test_op_handler_chain_registration()  -> !hex.chain {
  %ch0 = hex.new.chain
  %ch1 = hex.call @failed_get_op_handler(%ch0) : (!hex.chain) -> !hex.chain
  %ch2 = hex.call @register_op_handler_chain(%ch1) : (!hex.chain) -> !hex.chain
  %ch3 = hex.call @get_op_handler(%ch2) : (!hex.chain) -> !hex.chain
  hex.return %ch3 : !hex.chain
}
