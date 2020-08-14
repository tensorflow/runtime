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

// CHECK: --- Running 'tile_f32'
func @tile_f32() attributes {tfrt.sync} {
  %operand_0 = "tfrt_dht_sync.create_dense_tensor.f32"()
    { shape = [2, 3], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32] } : () -> !t.tensor

  %operand_1 = "tfrt_dht_sync.create_dense_tensor.i32"()
    { shape = [2], values = [2 : i32, 2 : i32] } : () -> !t.tensor

  %result = tfrt_dht_sync.create_uninitialized_tensor.f32.1 [4: i64, 6: i64]
  "tf_sync.Tile.f32"(%operand_0, %operand_1, %result) : (!t.tensor, !t.tensor, !t.tensor) -> ()

  // CHECK: DenseHostTensor dtype = F32, shape = [4, 6]
  // CHECK-SAME: values = [1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}

// CHECK: --- Running 'tile_string'
func @tile_string() attributes {tfrt.sync} {
  %operand_0 = "tfrt_sht_sync.create_tensor"()
    {shape = [2, 3], value = ["a", "b", "c", "d", "e", "f"]} : () -> !t.tensor

  %result = "tfrt_sht_sync.create_uninitialized_tensor"()
    {shape = [4, 6]} : () -> !t.tensor
  "tf_sync.Tile.string"(%operand_0, %result) : (!t.tensor, !t.tensor) -> ()

  // CHECK: StringHostTensor shape = [4, 6]
  // CHECK-SAME: values = ["a", "b", "c", "a", "b", "c", "d", "e", "f", "d", "e", "f", "a", "b", "c", "a", "b", "c", "d", "e", "f", "d", "e", "f"]
  tfrt_dht_sync.print_tensor %result

  tfrt.return
}
