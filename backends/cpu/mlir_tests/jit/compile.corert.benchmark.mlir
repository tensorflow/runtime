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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu $(bef_name %s) | FileCheck %s --dump-input=always

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

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'BM_compiled_add_f32'
func @BM_compiled_add_f32() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Allocate and initialize input tensor.
  %input_tensor = tfrt_dht.create_uninitialized_tensor.f32.2 [1024 : i64, 1024 : i64]
  %input_ready = tfrt_dht.fill_tensor_with_constant.f32 %input_tensor, %ch0 1.0 : f32
  %input = "corert.ht_to_tensorhandle" (%input_tensor, %input_ready):
    (!t.tensor, !tfrt.chain) -> !corert.tensorhandle

  // Output shape is the same as the input shape.
  %out_shape = "corert.tensorhandle_to_shape"(%input, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !ts.shape

  // Compile simple addition implemented as a Linalg generic operation.
  %compilation_result = cpurt.compile { kernel = @add_f32_kernel::@main }

  // Run compiled kernel benchmark.
  tfrt_test.benchmark "BM_compiled_add_f32"(
      %compilation_result : !cpurt.compilation_result,
      %input              : !corert.tensorhandle,
      %out_shape          : !ts.shape
  )
  duration_secs = 3, max_count = 100000, num_warmup_runs = 10
  {
    %result = cpurt.corert.execute %compilation_result
           (%input : !corert.tensorhandle)
           (%out_shape : !ts.shape) -> !corert.tensorhandle
    tfrt.return %result : !corert.tensorhandle
  }

  tfrt.return
}
