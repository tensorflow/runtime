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

// RUN: bef_executor %s.bef | FileCheck %s --dump-input=always

module @custom_call attributes { tfrt.compiled } {
  func.func private @noop.cc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                             %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>)
    attributes { rt.dynamic, rt.custom_call = "testlib.noop" }

  func.func @main(%arg0: memref<?x?xf32>)  {

    // Custom call is noop, we are only testing arguments and attributes
    // encoding/decoding performance.
    func.call @noop.cc(%arg0, %arg0, %arg0, %arg0) {
        str = "test string attribute encoding performance",
        f32 = 1.0 : f32,
        f64 = 2.0 : f64
      }
      : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,
         memref<?x?xf32>) -> ()

    func.return
  }
}

module @direct_custom_call attributes { tfrt.compiled } {
  func.func private @noop.cc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                             %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>)
    attributes { rt.custom_call = "testlib.noop" }

  func.func @main(%arg0: memref<?x?xf32>)  {

    // Custom call is noop, we only testing arguments and attributes
    // encoding/decoding performance.
    func.call @noop.cc(%arg0, %arg0, %arg0, %arg0) {
        str = "test string attribute encoding performance",
        f32 = 1.0 : f32,
        f64 = 2.0 : f64
      }
      : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,
         memref<?x?xf32>) -> ()

    func.return
  }
}

// CHECK: --- Running 'BM_compiled_custom_call'
func.func @BM_compiled_custom_call() {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  %executable = jitrt.compile { kernel = @custom_call::@main }

  // Run compiled kernel benchmark.
  tfrt_test.benchmark "BM_compiled_custom_call"(
      %executable  : !jitrt.jit_executable,
      %input_ready : !tfrt.chain,
      %input       : !t.tensor
  )
  duration_secs = 3, max_count = 1000000, num_warmup_runs = 10
  {
    jitrt.execute %executable[%input_ready](%input) : (!t.tensor) -> ()
    tfrt.return %input : !t.tensor
  }

  tfrt.return
}

// CHECK: --- Running 'BM_compiled_direct_custom_call'
func.func @BM_compiled_direct_custom_call() {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  %executable = jitrt.compile { kernel = @direct_custom_call::@main }

  // Run compiled kernel benchmark.
  tfrt_test.benchmark "BM_compiled_direct_custom_call"(
      %executable  : !jitrt.jit_executable,
      %input_ready : !tfrt.chain,
      %input       : !t.tensor
  )
  duration_secs = 3, max_count = 1000000, num_warmup_runs = 10
  {
    jitrt.execute %executable[%input_ready](%input) : (!t.tensor) -> ()
    tfrt.return %input : !t.tensor
  }

  tfrt.return
}
