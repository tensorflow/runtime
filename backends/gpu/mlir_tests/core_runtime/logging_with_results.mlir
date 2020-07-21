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

// RUN: bef_executor -devices="cpu:sync_logging|cpu,gpu:sync_logging|gpu"  $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail

// TODO(jingdong): Merge this file into logging.mlir after we fix the device creation process to allow more than one instances of a device type in a process.

// CHECK-LABEL: --- Running 'test_logger_cpu_log_results'
func @test_logger_cpu_log_results() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // CHECK: [0] dispatch 'tfrt_test.create_dense_tensor' 0 arguments, 1 result, OpAttrs contains 2 entries:
  // CHECK:  'shape' type=I64 value=[5]
  // CHECK:  'values' type=I32 value=[1, 2, 3, 4, 5]
  // CHECK: Inputs for [0]: 'tfrt_test.create_dense_tensor':
  // CHECK: Outputs for [0]: 'tfrt_test.create_dense_tensor':
  // CHECK:   Output for [0] tensor 0: DenseHostTensor dtype = I32, shape = [5], values = [1, 2, 3, 4, 5]

  %a_handle = corert.executeop(%cpu)
      "tfrt_test.create_dense_tensor"() { shape = [5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  // CHECK: [1] dispatch 'tfrt_test.odd_collector' 1 argument, 1 result, no attributes
  // CHECK: Inputs for [1]: 'tfrt_test.odd_collector':
  // CHECK:   Input for [1] tensor 0: DenseHostTensor dtype = I32, shape = [5], values = [1, 2, 3, 4, 5]
  // CHECK: Outputs for [1]: 'tfrt_test.odd_collector':
  // CHECK:   Output for [1] tensor 0: DenseHostTensor dtype = I32, shape = [3], values = [1, 3, 5]

  %b_handle = corert.executeop(%cpu) "tfrt_test.odd_collector"(%a_handle) : 1

   // CHECK: DenseHostTensor dtype = I32, shape = [3], values = [1, 3, 5]
  %ch3 = "corert.print_tensorhandle"(%b_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_logger_gpu_log_results'
func @test_logger_gpu_log_results() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu = corert.get_op_handler %ch0 "gpu"

  // CHECK: [0] dispatch 'tfrt_test.create_dense_tensor' 0 arguments, 1 result, OpAttrs contains 2 entries:
  // CHECK:  'shape' type=I64 value=[5]
  // CHECK:  'values' type=I32 value=[1, 2, 3, 4, 5]
  // CHECK: Inputs for [0]: 'tfrt_test.create_dense_tensor':
  // CHECK: Outputs for [0]: 'tfrt_test.create_dense_tensor':
  // CHECK:   Output for [0] tensor 0: DenseHostTensor dtype = I32, shape = [5], values = [1, 2, 3, 4, 5]
  %a_handle = corert.executeop(%gpu)
      "tfrt_test.create_dense_tensor"() { shape = [5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  // CHECK: DenseGpuTensor<dtype=I32, shape=[5], pointer=0x{{[0-9a-f]+}} (CUDA)
  %ch1 = "corert.print_tensorhandle"(%a_handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}
