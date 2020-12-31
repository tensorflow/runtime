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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running 'BM_compiled_add_f32'
func @BM_compiled_add_f32() {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Allocate uninitialized output tensor.
  %output = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]

  // Compile simple addition implemented as a Linalg generic operation.
  %compilation_result = cpurt.compile { mlir_module =
    "#map0 = affine_map<(d0, d1) -> (d0, d1)>\0A\0Afunc @main(%input: memref<?x?xf32>, %output: memref<?x?xf32>) -> !async.token {\0A\0A  %token = async.execute {\0A    linalg.generic {\0A      indexing_maps = [#map0, #map0],\0A      iterator_types = [\"parallel\", \"parallel\"]\0A    }\0A      ins(%input: memref<?x?xf32>)\0A      outs(%output : memref<?x?xf32>)\0A    {\0A      ^bb0(%in: f32, %out: f32):\0A        %0 = addf %in, %in : f32\0A        linalg.yield %0 : f32\0A    }\0A async.yield  }\0A  return %token : !async.token\0A}\0A"
  }

  // Run compiled kernel benchmark.
  tfrt_test.benchmark "BM_compiled_add_f32"(
      %compilation_result : !cpurt.compilation_result,
      %input_ready        : !tfrt.chain,
      %input              : !t.tensor,
      %output             : !t.tensor
  )
  duration_secs = 3, max_count = 100000, num_warmup_runs = 10
  {
    %executed = cpurt.execute %compilation_result[%input_ready]
                (%input, %output) : !t.tensor, !t.tensor
    tfrt.return %executed : !tfrt.chain
  }

  tfrt.return
}
