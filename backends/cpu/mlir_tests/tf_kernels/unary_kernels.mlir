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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'unary_ops'
func @unary_ops() attributes {tfrt.sync} {
  %a = "tfrt_test.sync.const_dense_attr"() {value = dense<[1.0, 2.5, 3.0, 4.5, 5.0]> : tensor<5xf32>} : () -> !t.tensor

  %b = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [5 : i64]

  "tf_sync.Log.f32"(%a, %b) : (!t.tensor, !t.tensor)->()

  // CHECK: dtype = F32, shape = [5], values = [0.000000e+00, 9.162908e-01, 1.098612e+00, 1.504077e+00, 1.609438e+00]
  tfrt_dht_sync.print_tensor %b

  "tf_sync.Log1p.f32"(%a, %b) : (!t.tensor, !t.tensor)->()

  // CHECK: dtype = F32, shape = [5], values = [6.931472e-01, 1.252763e+00, 1.386294e+00, 1.704748e+00, 1.791759e+00]
  tfrt_dht_sync.print_tensor %b

  %c = "tfrt_test.sync.const_dense_attr"() {value = dense<[2.0, 2.5, 3.0, 4.5, 5.0]> : tensor<5xf32>} : () -> !t.tensor
  "tf_sync.Rsqrt.f32"(%c, %b) : (!t.tensor, !t.tensor)->()

  // CHECK: dtype = F32, shape = [5], values = [7.071067e-01, 6.324555e-01, 5.773503e-01, 4.714045e-01, 4.472136e-01]
  tfrt_dht_sync.print_tensor %b

  %d = "tfrt_test.sync.const_dense_attr"() {value = dense<[1.0, 2.5, 3.0, 4.5, 6.0]> : tensor<5xf32>} : () -> !t.tensor
  "tf_sync.Sigmoid.f32"(%d, %b) : (!t.tensor, !t.tensor)->()

  // CHECK: dtype = F32, shape = [5], values = [7.310586e-01, 9.241418e-01, 9.525741e-01, 9.890131e-01, 9.975274e-01]
  tfrt_dht_sync.print_tensor %b

  tfrt.return
}
