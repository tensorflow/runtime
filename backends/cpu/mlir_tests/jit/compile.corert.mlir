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

// CHECK: --- Running 'compiled_add_f32'
func @compiled_add_f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1

  // Output shape is the same as the input shape.
  %out_shape = "corert.tensorhandle_to_shape"(%operand, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !ts.shape

  // Compile simple addition implemented as a Linalg generic operation.
  %compilation_result = cpurt.compile { kernel = @add_f32_kernel::@main }

  // Execute compiled kernel with tensor handle operands.
  %result = cpurt.corert.execute %compilation_result
           (%operand : !corert.tensorhandle)
           (%out_shape : !ts.shape) -> !corert.tensorhandle

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3]
  // CHECK-SAME: values = [-2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print"(%result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
