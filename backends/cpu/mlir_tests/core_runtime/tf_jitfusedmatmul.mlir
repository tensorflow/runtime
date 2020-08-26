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

// RUN: bef_executor -devices=cpu $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running 'fusedMatMul_AddOne_f32'
func @fusedMatMul_AddOne_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3, 2], values = [0.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu)
      "tf._JitFusedMatMul"(%operand_0, %operand_1)
      { mlir_blob = "func @compute(%output_block : memref<?x?xf32>, %row_offset : i64, %col_offset : i64) {\0A    %c0 = constant 0 : index\0A    %c1 = constant 1 : index\0A    %one = constant 1.0 : f32\0A\0A    %d0 = dim %output_block, %c0 : memref<?x?xf32>\0A    %d1 = dim %output_block, %c1 : memref<?x?xf32>\0A\0A    scf.for %i0 = %c0 to %d0 step %c1 {\0A      scf.for %i1 = %c0 to %d1 step %c1 {\0A        %0 = load %output_block[%i0, %i1] : memref<?x?xf32>\0A        %1 = addf %0, %one : f32\0A        store %1, %output_block[%i0, %i1] : memref<?x?xf32>\0A      }\0A    }\0A\0A    return\0A  }",
        transpose_a = false, transpose_b = false } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [0.000000e+00, -1.500000e+00, 9.000000e+00, 1.200000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'fusedMatMul_BiasAdd_f32'
func @fusedMatMul_BiasAdd_f32() -> !tfrt.chain{
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %operand_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1
  %operand_1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [3, 2], values = [0.0 : f32, 1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1
  %operand_2 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2], values = [1.0 : f32, 2.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu)
      "tf._JitFusedMatMul"(%operand_0, %operand_1, %operand_2)
      { mlir_blob = "func @compute(%output_block : memref<?x?xf32>,\0A                %row_offset : i64, %col_offset : i64,\0A                %bias : memref<?xf32>) {\0A    %c0 = constant 0 : index\0A    %c1 = constant 1 : index\0A    %one = constant 1.0 : f32\0A\0A    %d0 = dim %output_block, %c0 : memref<?x?xf32>\0A    %d1 = dim %output_block, %c1 : memref<?x?xf32>\0A\0A    // Inner dim offset == bias offset.\0A    %bias_offset = index_cast %col_offset : i64 to index\0A\0A    scf.for %i0 = %c0 to %d0 step %c1 {\0A      scf.for %i1 = %c0 to %d1 step %c1 {\0A        %0 = addi %bias_offset, %i1 : index\0A        %1 = load %output_block[%i0, %i1] : memref<?x?xf32>\0A        %2 = load %bias[%0] : memref<?xf32>\0A        %3 = addf %1, %2 : f32\0A        store %3, %output_block[%i0, %i1] : memref<?x?xf32>\0A      }\0A    }\0A\0A    return\0A  }",
        transpose_a = false, transpose_b = false } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2]
  // CHECK-SAME: values = [0.000000e+00, -5.000000e-01, 9.000000e+00, 1.300000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
