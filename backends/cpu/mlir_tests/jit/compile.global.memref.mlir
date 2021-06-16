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

// RUN: bef_executor $(bef_name %s)                          | FileCheck %s
// RUN: bef_executor $(bef_name %s) --work_queue_type=mstd:8 | FileCheck %s

module @kernels attributes { tfrt.compiled } {

  memref.global "private" @const : memref<2x2xf32> =
    dense<[[0.0, 1.0], [2.0, 3.0]]>

  func @main() -> memref<2x2xf32> {
    %0 = memref.get_global @const : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }
}

// CHECK: --- Running 'global_memref_to_tensor'
func @global_memref_to_tensor() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %executable = cpurt.compile { kernel = @kernels::@main }
  %0 = cpurt.execute %executable[%ch0]() : () -> !t.tensor
  // CHECK:      DenseHostTensor dtype = f32, shape = [2, 2]
  // CHECK-SAME: values = [0.0{{.*}}, 1.0{{.*}}, 2.0{{.*}}, 3.0{{.*}}]
  %printed = tfrt_dht.print_tensor %0, %ch0

  tfrt.return %printed : !tfrt.chain
}
