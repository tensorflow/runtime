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

// RUN: bef_executor %s.bef | FileCheck %s

// CHECK: --- Running 'const_kernel'
func @const_kernel() attributes {tfrt.sync} {
  %a = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [5 : i64]
  "tf_sync.Const"(%a) {value = dense<[1.0, 2.5, 3.0, 4.5, 5.0]> : tensor<5xf32>} : (!t.tensor) -> ()

  // CHECK: shape = [5], values = [1.000000e+00, 2.500000e+00, 3.000000e+00, 4.500000e+00, 5.000000e+00]
  tfrt_dht_sync.print_tensor %a

  tfrt.return
}
