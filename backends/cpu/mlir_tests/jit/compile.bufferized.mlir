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

// XFAIL: *
// RUN: bef_executor %s.bef                          | FileCheck %s
// RUN: bef_executor %s.bef --work_queue_type=mstd:8 | FileCheck %s

module @kernels attributes { tfrt.compiled } {
  // Bufferized kernel that takes input and output buffers, and returns a token
  // that signals when output is ready.
  func @main(%input: memref<?x?xf32>, %output: memref<?x?xf32>)
                   -> !async.token {
    %token = async.execute {
      linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"] }
      ins(%input: memref<?x?xf32>) outs(%output : memref<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %0 = addf %in, %in : f32
          linalg.yield %0 : f32
      }
      async.yield
    }
    return %token : !async.token
  }
}

// CHECK: --- Running 'compiled_add_f32_buffers'
func @compiled_add_f32_buffers() {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Allocate and initialize output tensor (to suppress msan warnings).
  %output = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %output, %ch1 1.0 : f32

  // Allocate and initialize expected tensor.
  %expected = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch3 = tfrt_dht.fill_tensor_with_constant.f32 %expected, %ch2 2.0 : f32

  // Compile simple addition implemented as a Linalg generic operation.
  %executable = cpurt.compile { kernel = @kernels::@main }

  // Execute compiled kernel with tensor operands.
  %executed = cpurt.execute %executable[%ch3](%input, %output)
              : (!t.tensor, !t.tensor) -> !tfrt.chain

  // Wait for the execution completion and compare result with expected.
  %cmp, %cmp_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output, %executed)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  tfrt.print.i1 %cmp, %cmp_ch

  tfrt.return
}
