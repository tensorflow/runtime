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

// CHECK-LABEL: --- Running 'test_softmax_f32'
func @test_softmax_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.Softmax"(%operand_0) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: [2.689414e-01, 7.310586e-01, 2.689414e-01, 7.310586e-01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_log_softmax_f32'
func @test_log_softmax_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.LogSoftmax"(%operand_0) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: [-1.313262e+00, -3.132617e-01, -1.313262e+00, -3.132617e-01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
