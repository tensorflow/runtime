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

// RUN: bef_executor %s.bef 2>&1 | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Not running 'register_op_handlers' because it has arguments.
func @register_op_handlers(%ch0: !tfrt.chain) -> !tfrt.chain {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  %op_handler = "corert.create_logging_op_handler"(%cpu) {sync_log_results=1} : (!corert.ophandler) -> !corert.ophandler
  %ch = corert.register_op_handler %op_handler "sync_logging"
  tfrt.return %ch : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_logger'
func @test_logger() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @register_op_handlers(%ch0) : (!tfrt.chain) -> !tfrt.chain
  %log = corert.get_op_handler %ch1 "sync_logging"

  // CHECK: [0] dispatch 'tfrt_test.create_dense_tensor' 0 arguments, 1 result, OpAttrs contains 2 entries:
  // CHECK:  'shape' type=I64 value=[5]
  // CHECK:  'values' type=I32 value=[1, 2, 3, 4, 5]
  %a_handle = corert.executeop(%log)
      "tfrt_test.create_dense_tensor"() { shape = [5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  // CHECK: [1] dispatch 'tfrt_test.odd_collector' 1 argument, 1 result, no attributes
  %b_handle = corert.executeop(%log) "tfrt_test.odd_collector"(%a_handle) : 1

  // CHECK: DenseHostTensor dtype = i32, shape = [3], values = [1, 3, 5]
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}
