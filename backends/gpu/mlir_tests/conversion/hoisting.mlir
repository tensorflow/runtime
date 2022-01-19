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

// RUN: tfrt_gpu_opt %s -gpu-tfrt-hoisting | FileCheck %s

// CHECK-LABEL: func @tfrt_gpu.blas.create
// CHECK-SAME: (%arg0: !tfrt_gpu.context) -> !tfrt_gpu.blas.handle {
// CHECK:   %0 = tfrt_gpu.blas.create %arg0
// CHECK:   tfrt.return %0 : !tfrt_gpu.blas.handle
// CHECK: }

// CHECK-LABEL: @gpu_hoist_create_blas_handle
func @gpu_hoist_create_blas_handle(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> !tfrt.chain {
  // CHECK: %[[context:.*]] = tfrt_gpu.stream.get_context %arg1
  %context = tfrt_gpu.stream.get_context %arg1
  // CHECK: tfrt.once @tfrt_gpu.blas.create(%[[context]])
  // CHECK-SAME: : (!tfrt_gpu.context) -> (!tfrt_gpu.blas.handle)
  %blas = tfrt_gpu.blas.create %context
  tfrt.return %arg0 : !tfrt.chain
}

// CHECK-LABEL: func @tfrt_gpu.dnn.create
// CHECK-SAME: (%arg0: !tfrt_gpu.context) -> !tfrt_gpu.dnn.handle {
// CHECK:   %0 = tfrt_gpu.dnn.create %arg0
// CHECK:   tfrt.return %0 : !tfrt_gpu.dnn.handle
// CHECK: }

// CHECK-LABEL: @gpu_hoist_create_dnn_handle
func @gpu_hoist_create_dnn_handle(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> !tfrt.chain {
  // CHECK: %[[context:.*]] = tfrt_gpu.stream.get_context %arg1
  %context = tfrt_gpu.stream.get_context %arg1
  // CHECK: tfrt.once @tfrt_gpu.dnn.create(%[[context]])
  // CHECK-SAME: : (!tfrt_gpu.context) -> (!tfrt_gpu.dnn.handle)
  %dnn = tfrt_gpu.dnn.create %context
  tfrt.return %arg0 : !tfrt.chain
}

// CHECK-LABEL: func @tfrt_gpu.solver.create
// CHECK-SAME: (%arg0: !tfrt_gpu.context) -> !tfrt_gpu.solver.handle {
// CHECK:   %0 = tfrt_gpu.solver.create %arg0
// CHECK:   tfrt.return %0 : !tfrt_gpu.solver.handle
// CHECK: }

// CHECK-LABEL: @gpu_hoist_create_solver_handle
func @gpu_hoist_create_solver_handle(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> !tfrt.chain {
  // CHECK: %[[context:.*]] = tfrt_gpu.stream.get_context %arg1
  %context = tfrt_gpu.stream.get_context %arg1
  // CHECK: tfrt.once @tfrt_gpu.solver.create(%[[context]])
  // CHECK-SAME: : (!tfrt_gpu.context) -> (!tfrt_gpu.solver.handle)
  %solver = tfrt_gpu.solver.create %context
  tfrt.return %arg0 : !tfrt.chain
}
