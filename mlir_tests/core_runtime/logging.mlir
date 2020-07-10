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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices="sync_logging|cpu" 2>&1 | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'test_logger'
func @test_logger() -> !hex.chain {
  %ch0 = hex.new.chain
  %log = corert.get_op_handler %ch0 "sync_logging"

  // CHECK: [0] dispatch 'tfrt_test.create_dense_tensor' 0 arguments, 1 result, OpAttrs contains 2 entries:
  // CHECK:  'shape' type=I64 value=[5]
  // CHECK:  'values' type=I32 value=[1, 2, 3, 4, 5]
  %a_handle = corert.executeop(%log)
      "tfrt_test.create_dense_tensor"() { shape = [5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  // CHECK: [1] dispatch 'tfrt_test.odd_collector' 1 argument, 1 result, no attributes
  %b_handle = corert.executeop(%log) "tfrt_test.odd_collector"(%a_handle) : 1

  // CHECK: DenseHostTensor dtype = I32, shape = [3], values = [1, 3, 5]
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch3 : !hex.chain
}
