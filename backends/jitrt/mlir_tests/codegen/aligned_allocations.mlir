// Copyright 2022 The TensorFlow Runtime Authors
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

// RUN: jitrt_opt %s --codegen-aligned-allocations | FileCheck %s
// RUN: jitrt_opt %s --codegen-aligned-allocations=alignment=16 \
// RUN: | FileCheck --check-prefix=ALIGN16 %s

// CHECK-LABEL: @aligned_alloc
// ALIGN16-LABEL: @aligned_alloc
func @aligned_alloc(%arg0: index) -> memref<?xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 64 : i64} : memref<?xf32>
  // CHECK: return %[[ALLOC]]
  // ALIGN16: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 32 : i64} : memref<?xf32>
  // ALIGN16: return %[[ALLOC]]
  %0 = memref.alloc(%arg0) { alignment = 32 : i64 } : memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: @unaligned_alloc
// ALIGN16-LABEL: @unaligned_alloc
func @unaligned_alloc(%arg0: index) -> memref<?xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 64 : i64} : memref<?xf32>
  // CHECK: return %[[ALLOC]]
  // ALIGN16: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 16 : i64} : memref<?xf32>
  // ALIGN16: return %[[ALLOC]]
  %0 = memref.alloc(%arg0) : memref<?xf32>
  return %0 : memref<?xf32>
}

