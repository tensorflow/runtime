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

// Define A+B (f32) kernel in Linalg dialect.
module @add_f32_kernel attributes { tfrt.compiled } {
  func @main(%input: memref<?x?xf32>, %output: memref<?x?xf32>) -> !async.token {
    %token = async.execute {
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
        ins(%input: memref<?x?xf32>)
        outs(%output : memref<?x?xf32>)
      {
        ^bb0(%in: f32, %out: f32):
          %0 = addf %in, %in : f32
          linalg.yield %0 : f32
      }
      async.yield
    }
    return %token : !async.token
  }
}

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
  %compilation_result = cpurt.compile { kernel = @add_f32_kernel::@main }

  // Execute compiled kernel with tensor operands.
  %executed = cpurt.execute %compilation_result[%input_ready]
                (%input, %output : !t.tensor, !t.tensor)

  // Wait for the execution completion and compare result with expected.
  %cmp, %cmp_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output, %executed)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  tfrt.print.i1 %cmp, %cmp_ch

  tfrt.return
}
