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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu  \
// RUN:   %s.bef | FileCheck %s

// RUN: bef_executor --test_init_function=register_op_handlers_cpu \
// RUN:              --work_queue_type=mstd:8                      \
// RUN:   %s.bef | FileCheck %s

module @kernels attributes { tfrt.compiled } {
  func @main(%input: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %input, %c0 : memref<?x?xf32>
    %1 = memref.dim %input, %c1 : memref<?x?xf32>
    %output = memref.alloc(%0, %1) : memref<?x?xf32>

    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
      ins(%input: memref<?x?xf32>)
      outs(%output : memref<?x?xf32>)
    {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.addf %in, %in : f32
        linalg.yield %2 : f32
    }

    return %output : memref<?x?xf32>
  }
}

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null)
         : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'compiled_add_f32_corert'
func @compiled_add_f32_corert() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { values = [-1.0: f32, -0.5: f32, 0.0: f32, 0.5: f32, 1.0: f32, 1.5: f32],
      shape = [2, 3] } : 1

  // Compile simple addition implemented as a Linalg generic operation.
  %executable = cpurt.corert.compile { kernel = @kernels::@main }

  // Execute compiled kernel with tensor handle operands.
  %result = cpurt.corert.execute %executable (%operand)
              : (!corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: DenseHostTensor dtype = f32, shape = [2, 3]
  // CHECK-SAME: values =
  // CHECK-SAME: -2.0{{.*}}, -1.0{{.*}}, 0.0{{.*}}, 1.0{{.*}}, 2.0{{.*}}, 3.0{{.*}}
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print"(%result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
