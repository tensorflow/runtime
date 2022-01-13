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

// RUN: jitrt_opt %s -split-input-file -cost-driven-async-parallel-for="legacy-behavior=false" -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @loop1d
func @loop1d(%arg: memref<?xf32>) {
  %cst = arith.constant 42.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg, %c0 : memref<?xf32>
  scf.parallel (%i) = (%c0) to (%d0) step (%c1) {
    memref.store %cst, %arg[%i] : memref<?xf32>
    // expected-remark@above {{ramCost: { %c4 = arith.constant 4 : index } cpuCost: { %c0_0 = arith.constant 0 : index }}}
    scf.yield
    // expected-remark@above {{ramCost: { %c0_1 = arith.constant 0 : index } cpuCost: { %c0_2 = arith.constant 0 : index }}}
  }
  // expected-remark@-6 {{ramCost: { %c0_3 = arith.constant 0 : index } cpuCost: { %c0_4 = arith.constant 0 : index }}}
  // expected-remark@-7 {{ramCost: { %1 = arith.addi %c0_3, %c4 : index } cpuCost: { %2 = arith.addi %c0_4, %c0_0 : index }}}
  // expected-remark@-8 {{ramCost: { %3 = arith.addi %1, %c0_1 : index } cpuCost: { %4 = arith.addi %2, %c0_2 : index }}}
  return
}

// -----

// CHECK-LABEL: func @loop2d
func @loop2d(%arg: memref<?x?xf32>) {
  %cst = arith.constant 42.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg, %c1 : memref<?x?xf32>
  scf.parallel (%i) = (%c0) to (%d0) step (%c1) {
    scf.for %j = %c0 to %d1 step %c1 {
      memref.store %cst, %arg[%i, %j] : memref<?x?xf32>
      // expected-remark@above {{ramCost: { %c4 = arith.constant 4 : index } cpuCost: { %c0_0 = arith.constant 0 : index }}}
      scf.yield
      // expected-remark@above {{ramCost: { %c0_1 = arith.constant 0 : index } cpuCost: { %c0_2 = arith.constant 0 : index }}}
    }
    // expected-remark@-6  {{ramCost: { %c0_3 = arith.constant 0 : index } cpuCost: { %c0_4 = arith.constant 0 : index }}}
    // expected-remark@-7  {{ramCost: { %2 = arith.addi %c0_3, %c4 : index } cpuCost: { %3 = arith.addi %c0_4, %c0_0 : index }}}
    // expected-remark@-8  {{ramCost: { %4 = arith.addi %2, %c0_1 : index } cpuCost: { %5 = arith.addi %3, %c0_2 : index }}}
    // expected-remark@-9  {{iterations: %7 = arith.ceildivsi %6, %c1 : index}}
    // expected-remark@-10 {{ramCost: { %8 = arith.muli %4, %7 : index } cpuCost: { %9 = arith.muli %5, %7 : index }}}
    scf.yield
    // expected-remark@above {{ramCost: { %c0_5 = arith.constant 0 : index } cpuCost: { %c0_6 = arith.constant 0 : index }}}
  }
  // expected-remark@-15  {{ramCost: { %c0_7 = arith.constant 0 : index } cpuCost: { %c0_8 = arith.constant 0 : index }}}
  // expected-remark@-16  {{ramCost: { %10 = arith.addi %c0_7, %8 : index } cpuCost: { %11 = arith.addi %c0_8, %9 : index }}}
  // expected-remark@-17  {{ramCost: { %12 = arith.addi %10, %c0_5 : index } cpuCost: { %13 = arith.addi %11, %c0_6 : index }}}
  return
}
