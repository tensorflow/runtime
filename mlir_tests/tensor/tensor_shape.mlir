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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: tfrt_opt %s | tfrt_opt

// CHECK-LABEL: --- Running 'basic_shape'
func @basic_shape() {
  // Normal shape.
  %a = ts.build_shape [1 : i64, 57 : i64, 92 : i64]

  // CHECK: shape = [1, 57, 92]
  ts.print_shape %a

  // Zero-D shape.
  %b = ts.build_shape []

  // CHECK: shape = []
  ts.print_shape %b

  // 32-bit shape.
  %c = ts.build_shape [ 65537 : i64 ]

  // CHECK: shape = [65537]
  ts.print_shape %c

  // High rank shape.
  %d = ts.build_shape [
        0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64,
        5 : i64, 6 : i64, 7 : i64, 8 : i64, 9 : i64 ]

  // CHECK: shape = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  ts.print_shape %d

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_shape_equal'
func @tensor_shape_equal() {
  %ch0 = tfrt.new.chain

  // Normal shape.
  %a1 = ts.build_shape [1 : i64, 57 : i64, 92 : i64]
  %a2 = ts.build_shape [1 : i64, 57 : i64, 92 : i64]
  %aa = ts.equal_shape %a1, %a2
  // CHECK: int1 = 1
  tfrt.print.i1 %aa, %ch0

  // Zero-D shape.
  %b1 = ts.build_shape []
  %b2 = ts.build_shape []
  %bb = ts.equal_shape %b1, %b2
  // CHECK: int1 = 1
  tfrt.print.i1 %bb, %ch0

  // 32-bit shape.
  %c1 = ts.build_shape [ 65537 : i64 ]
  %c2 = ts.build_shape [ 65537 : i64 ]
  %cc = ts.equal_shape %c1, %c2
  // CHECK: int1 = 1
  tfrt.print.i1 %cc, %ch0

  // High rank shape.
  %d1 = ts.build_shape [
         0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64,
         5 : i64, 6 : i64, 7 : i64, 8 : i64, 9 : i64 ]
  %d2 = ts.build_shape [
         0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64,
         5 : i64, 6 : i64, 7 : i64, 8 : i64, 9 : i64 ]
  %dd = ts.equal_shape %d1, %d2
  // CHECK: int1 = 1
  tfrt.print.i1 %dd, %ch0

  // Inequality.
  %ab = ts.equal_shape %a1, %b1
  // CHECK: int1 = 0
  tfrt.print.i1 %ab, %ch0

  %ac = ts.equal_shape %a1, %c1
  // CHECK: int1 = 0
  tfrt.print.i1 %ac, %ch0

  %ad = ts.equal_shape %a1, %d1
  // CHECK: int1 = 0
  tfrt.print.i1 %ad, %ch0

  %bc = ts.equal_shape %b1, %c1
  // CHECK: int1 = 0
  tfrt.print.i1 %bc, %ch0

  %bd = ts.equal_shape %b1, %d1
  // CHECK: int1 = 0
  tfrt.print.i1 %bd, %ch0

  %cd = ts.equal_shape %c1, %d1
  // CHECK: int1 = 0
  tfrt.print.i1 %cd, %ch0

  tfrt.return
}

// CHECK-LABEL: --- Running 'representation_edge_cases'
func @representation_edge_cases() {
  // Largest shape that fits in rep16.
  %a = ts.build_shape [
        65535 : i64, 65535 : i64, 65535 : i64, 65535 : i64,
        65535 : i64, 65535 : i64, 65535 : i64]

  // CHECK: shape = [65535, 65535, 65535, 65535, 65535, 65535, 65535]
  ts.print_shape %a

  // Rank too high for rep16.
  %b = ts.build_shape [
        65535 : i64, 65535 : i64, 65535 : i64, 65535 : i64,
        65535 : i64, 65535 : i64, 65535 : i64, 1 : i64]

  // CHECK: shape = [65535, 65535, 65535, 65535, 65535, 65535, 65535, 1]
  ts.print_shape %b

  // Last dimension too big for rep16.
  %c = ts.build_shape [
        65535 : i64, 65535 : i64, 65535 : i64, 65535 : i64,
        65535 : i64, 65535 : i64, 65536 : i64]

  // CHECK: shape = [65535, 65535, 65535, 65535, 65535, 65535, 65536]
  ts.print_shape %c

  // Largest shape that fits in rep32.
  %d = ts.build_shape [
        4294967295 : i64, 4294967295 : i64, 4294967295 : i64,
        65535 : i64]

  // CHECK: shape = [4294967295, 4294967295, 4294967295, 65535]
  ts.print_shape %d

  // Rank too high for rep32.
  %e = ts.build_shape [
        4294967295 : i64, 4294967295 : i64, 4294967295 : i64, 65535 : i64,
        1 : i64]

  // CHECK: shape = [4294967295, 4294967295, 4294967295, 65535, 1]
  ts.print_shape %e

  // Last dimension too big for rep32.
  %f = ts.build_shape [
        4294967295 : i64, 4294967295 : i64, 4294967295 : i64,
        65536 : i64]

  // CHECK: shape = [4294967295, 4294967295, 4294967295, 65536]
  ts.print_shape %f

  tfrt.return
}

// CHECK-LABEL: --- Running 'fixed_rank_shape'
func @fixed_rank_shape() {
  %a = ts.build_shape [1 : i64, 57 : i64, 92 : i64]
  %b = ts.as_fixed_rank_shape.3 %a

  // CHECK: fixed_rank_shape = [1, 57, 92]
  ts.print_fixed_rank_shape.3 %b

  tfrt.return
}

// CHECK-LABEL: --- Running 'ts_get_num_elements'
func @ts_get_num_elements() {
  %ch0 = tfrt.new.chain

  %a = ts.build_shape [1 : i64, 10 : i64, 10 : i64]
  %b = ts.get_num_elements %a

  // CHECK: int64 = 100
  %ch1 = tfrt.print.i64 %b, %ch0
  tfrt.return
}

// CHECK-LABEL: --- Running 'partial_tensor_shape'
func @partial_tensor_shape() {
  %a = ts.build_partial_shape [1 : i64, 57 : i64, 92 : i64]

  // CHECK: partial_tensor_shape = [1, 57, 92]
  ts.print_partial_shape %a

  %b = ts.to_shape %a

  // CHECK: shape = [1, 57, 92]
  ts.print_shape %b

  tfrt.return
}

// CHECK-LABEL: --- Running 'partial_tensor_shape_with_unknown_dim'
func @partial_tensor_shape_with_unknown_dim() {
  %a = ts.build_partial_shape [-1 : i64, 57 : i64, 92 : i64]

  // CHECK: partial_tensor_shape = [-1, 57, 92]
  ts.print_partial_shape %a

  tfrt.return
}

// CHECK-LABEL: --- Running 'partial_tensor_shape_to_tensor_shape_error'
func @partial_tensor_shape_to_tensor_shape_error() {
  %a = ts.build_partial_shape [-1 : i64, 57 : i64, -1 : i64]

  // CHECK: partial_tensor_shape = [-1, 57, -1]
  ts.print_partial_shape %a

  // expected-error @+1 {{runtime error: Unknown dimensions at following indices = [0, 2]}}
  %b = ts.to_shape %a

  tfrt.return
}

// CHECK-LABEL: --- Running 'partial_tensor_shape_unranked'
func @partial_tensor_shape_unranked() {
  %a = ts.build_unranked_partial_shape

  // CHECK: partial_tensor_shape = Unknown rank
  ts.print_partial_shape %a

  tfrt.return
}

// CHECK-LABEL: --- Running 'ts_to_partial_shape'
func @ts_to_partial_shape() {
  %a = ts.build_shape [1 : i64, 57 : i64, 92 : i64]

  // CHECK: shape = [1, 57, 92]
  ts.print_shape %a

  %b = ts.to_partial_shape %a

  // CHECK: shape = [1, 57, 92]
  ts.print_partial_shape %b

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_shape_type_check'
func @tensor_shape_type_check() {
  %a = "ts.build_shape"() { value = [1 : i64, 57 : i64, 92 : i64] } : () -> !ts.shape

  // CHECK: shape = [1, 57, 92]
  ts.print_shape %a

  %b = "ts.build_partial_shape"() { value = [-1 : i64, 57 : i64, -1 : i64] } : () -> !ts.partial_shape

  // CHECK: partial_tensor_shape = [-1, 57, -1]
  ts.print_partial_shape %b

  tfrt.return
}
