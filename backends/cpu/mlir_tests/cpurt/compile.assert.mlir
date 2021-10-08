// Copyright 2021 The TensorFlow Runtime Authors
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

// RUN: bef_executor %s.bef                          | FileCheck %s
// RUN: bef_executor %s.bef --work_queue_type=mstd:8 | FileCheck %s

module @kernels attributes { tfrt.compiled } {
  func @main(%input: memref<?xf32>) -> memref<?xf32> {
    %c0 = constant 0 : index
    %0 = memref.dim %input, %c0 : memref<?xf32>

    // This precondition is always false at run time.
    %check = cmpi eq, %0, %c0 : index
    assert %check, "Dimension 0 must have size 0"

    // We should never reach the memory allocation at run time.
    %output = memref.alloc(%0) : memref<?xf32>
    return %output : memref<?xf32>
  }
}

// CHECK: --- Running 'runtime_error'
func @runtime_error() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.1 [16 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  %executable = cpurt.compile { kernel = @kernels::@main }

  // expected-error @+1 {{Failed to execute the compiled kernel function}}
  %output = cpurt.execute %executable[%input_ready](%input)
              : (!t.tensor) -> (!t.tensor)

  tfrt.return %ch0 : !tfrt.chain
}
