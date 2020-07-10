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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'test_softmax_f32'
func @test_softmax_f32() {
  %ch0 = hex.new.chain

  %t1 = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 2 : i64] }
    : () -> !t.tensor

  %ch1 = "tfrt_dht.set_tensor_with_constant_values.f32"(%t1, %ch0)
    { values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] }
    : (!t.tensor, !hex.chain) -> !hex.chain

  %t2 = "tfrt_dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64, 2 : i64] }
    : () -> !t.tensor

  %ch2 = "eigen.softmax.f32"(%t1, %t2, %ch1) { log = false }
    : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: [2.689414e-01, 7.310586e-01, 2.689414e-01, 7.310586e-01]
  tfrt_dht.print_tensor %t2, %ch2
  hex.return
}

// CHECK-LABEL: --- Running 'test_log_softmax_f32'
func @test_log_softmax_f32() {
  %ch0 = hex.new.chain

  %t1 = "tfrt_dht.create_uninitialized_tensor.f32.4"()
    { shape = [2 : i64, 2 : i64] }
    : () -> !t.tensor

  %ch1 = "tfrt_dht.set_tensor_with_constant_values.f32"(%t1, %ch0)
    { values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32] }
    : (!t.tensor, !hex.chain) -> !hex.chain

  %t2 = "tfrt_dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64, 2 : i64] }
    : () -> !t.tensor

  %ch2 = "eigen.softmax.f32"(%t1, %t2, %ch1) { log = true }
    : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: [-1.313262e+00, -3.132617e-01, -1.313262e+00, -3.132617e-01]
  tfrt_dht.print_tensor %t2, %ch2
  hex.return
}
