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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu %s.bef | FileCheck %s

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_softmax_f32'
func @test_softmax_f32() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] }
    : () -> !t.tensor

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64, 2: i64]
  "tf_sync.Softmax.f32"(%operand_0, %result) : (!t.tensor, !t.tensor)->()

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: [2.689414e-01, 7.310586e-01, 2.689414e-01, 7.310586e-01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_log_softmax_f32'
func @test_log_softmax_f32() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 2], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] }
    : () -> !t.tensor

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [2: i64, 2: i64]
  "tf_sync.LogSoftmax.f32"(%operand_0, %result) : (!t.tensor, !t.tensor)->()

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: [-1.313262e+00, -3.132617e-01, -1.313262e+00, -3.132617e-01]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}
