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

// RUN: bef_executor $(bef_name %s)                          | FileCheck %s
// RUN: bef_executor $(bef_name %s) --work_queue_type=mstd:8 | FileCheck %s

// CHECK: --- Running 'compiled_add_f32'
func @compiled_add_f32() {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Allocate uninitialized output tensor.
  %output = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]

  // Allocate and initialzie expected tensor.
  %expected = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %expected_ready = tfrt_dht.fill_tensor_with_constant.f32 %expected, %ch0 2.0 : f32

  // Compile simple addition implemented as a Linalg generic operation.
  %compilation_result = cpurt.compile { mlir_module =
    "#map0 = affine_map<(d0, d1) -> (d0, d1)>\0A\0Afunc @main(%input: memref<?x?xf32>, %output: memref<?x?xf32>) -> !async.token {\0A\0A  %token = async.execute {\0A    linalg.generic {\0A      indexing_maps = [#map0, #map0],\0A      iterator_types = [\"parallel\", \"parallel\"]\0A    }\0A      ins(%input: memref<?x?xf32>)\0A      outs(%output : memref<?x?xf32>)\0A    {\0A      ^bb0(%in: f32, %out: f32):\0A        %0 = addf %in, %in : f32\0A        linalg.yield %0 : f32\0A    }\0A async.yield  }\0A  return %token : !async.token\0A}\0A"
  }

  // Execute compiled kernel with tensor operands.
  %executed = cpurt.execute %compilation_result[%input_ready](%input, %output) : !t.tensor, !t.tensor

  // Wait for the execution completion and compare result with expected.
  %cmp, %cmp_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output, %executed)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  tfrt.print.i1 %cmp, %cmp_ch

  tfrt.return
}
