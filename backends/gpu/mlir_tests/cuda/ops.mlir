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

// RUN: tfrt_gpu_opt %s | FileCheck %s

func @driver_ops() {
  // CHECK: %[[ordinal:.*]] = tfrt.constant.i32 0
  %ordinal = tfrt.constant.i32 0
  // CHECK: %[[device:.*]] = tfrt_gpu.device.get CUDA, %[[ordinal]]
  %device = tfrt_gpu.device.get CUDA, %ordinal
  // CHECK: %[[primary:.*]] = tfrt_gpu.context.primary %[[device]]
  %primary = tfrt_gpu.context.primary %device
  // CHECK: %[[context:.*]] = tfrt_gpu.context.create %[[device]]
  %context = tfrt_gpu.context.create %device
  // CHECK: %[[allocator:.*]] = tfrt_gpu.allocator.create %[[context]]
  %allocator = tfrt_gpu.allocator.create %context
  // CHECK: %[[stream:.*]] = tfrt_gpu.stream.create %[[context]]
  %stream = tfrt_gpu.stream.create %context
  // CHECK: %[[event:.*]] = tfrt_gpu.event.create %[[context]]
  %event = tfrt_gpu.event.create %context
  // CHECK: tfrt_gpu.stream.get_context %[[stream]]
  %_ctx = tfrt_gpu.stream.get_context %stream

  %ch0 = tfrt.new.chain
  // CHECK: tfrt_gpu.stream.synchronize %[[stream]], %{{.*}}
  %ch1 = tfrt_gpu.stream.synchronize %stream, %ch0
  // CHECK: tfrt_gpu.event.record %[[event]], %[[stream]], %{{.*}}
  %ch2 = tfrt_gpu.event.record %event, %stream, %ch1
  // CHECK: tfrt_gpu.stream.wait %[[stream]], %[[event]], %{{.*}}
  %ch3 = tfrt_gpu.stream.wait %stream, %event, %ch2
  // CHECK: tfrt_gpu.event.synchronize %[[event]], %{{.*}}
  %ch4 = tfrt_gpu.event.synchronize %event, %ch3

  // CHECK: %[[size:.*]] = tfrt.constant.i64 1024
  %size = tfrt.constant.i64 1024
  // CHECK: %[[buffer:.*]] = tfrt_gpu.mem.allocate %[[allocator]], %[[stream]], %[[size]], %{{.*}}
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch4

  %host_tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [256 : i64]
  // CHECK: %[[host_buffer:.*]]:2 = tfrt_dht.get_buffer
  %host_buffer:2 = tfrt_dht.get_buffer %host_tensor, %ch4
  // CHECK: %[[pinned_buffer:.*]] = tfrt_gpu.mem.register %[[context]], %[[host_buffer]]
  %pinned_buffer = tfrt_gpu.mem.register %context, %host_buffer
  // CHECK: %[[offset:.*]] = tfrt.constant.ui32 42
  %offset = tfrt.constant.ui32 42
  // CHECK: tfrt_gpu.mem.view %[[buffer]], %[[offset]]
  %view = tfrt_gpu.mem.view %buffer, %offset
  // CHECK: tfrt_gpu.mem.copy %[[host_buffer]]#0, %[[buffer]], %[[stream]], %{{.*}} : !ht.host_buffer, !tfrt_gpu.buffer
  %ch5 = tfrt_gpu.mem.copy %host_buffer#0, %buffer, %stream, %ch4 : !ht.host_buffer, !tfrt_gpu.buffer
  // CHECK: %[[value:.*]] = tfrt.constant.i32 13
  %value = tfrt.constant.i32 13
  // CHECK: tfrt_gpu.mem.set %[[buffer]], %[[value]], %[[stream]], %{{.*}}
  %ch6 = tfrt_gpu.mem.set %buffer, %value, %stream, %ch5 : !tfrt_gpu.buffer, i32

  // CHECK: %[[module:.*]] = tfrt_gpu.module.load %[[context]] {data = "foobar\00"}
  %module = tfrt_gpu.module.load %context {data = "foobar\00"}
  // CHECK: %[[global:.*]] = tfrt_gpu.module.get_global %[[module]] {name = "symbol"}
  %global = tfrt_gpu.module.get_global %module {name = "symbol"}
  // CHECK: %[[function:.*]] = tfrt_gpu.module.get_function %[[module]] {name = "kernel"}
  %function = tfrt_gpu.module.get_function %module {name = "kernel"}

  // CHECK: %[[dim:.*]] = tfrt.constant.ui32 1
  %dim = tfrt.constant.ui32 1
  // CHECK: %[[shmem:.*]] = tfrt.constant.ui32 0
  %shmem = tfrt.constant.ui32 0
  // CHECK: tfrt_gpu.function.launch %[[stream]], %[[function]],
  // CHECK-SAME: blocks in (%[[dim]], %[[dim]], %[[dim]]),
  // CHECK-SAME: threads in (%[[dim]], %[[dim]], %[[dim]]),
  // CHECK-SAME: %[[shmem]], %{{.*}},
  // CHECK-SAME: args(%[[buffer]]) : (!tfrt_gpu.buffer)
  %ch7 = tfrt_gpu.function.launch %stream, %function,
             blocks in (%dim, %dim, %dim),
             threads in (%dim, %dim, %dim),
             %shmem, %ch6,
             args(%buffer) : (!tfrt_gpu.buffer)

  tfrt.return
}

func @blas_ops() {
  // CHECK: %[[ordinal:.*]] = tfrt.constant.i32 0
  %ordinal = tfrt.constant.i32 0
  // CHECK: %[[device:.*]] = tfrt_gpu.device.get CUDA, %[[ordinal]]
  %device = tfrt_gpu.device.get CUDA, %ordinal
  // CHECK: %[[context:.*]] = tfrt_gpu.context.create %[[device]]
  %context = tfrt_gpu.context.create %device
  // CHECK: %[[allocator:.*]] = tfrt_gpu.allocator.create %[[context]]
  %allocator = tfrt_gpu.allocator.create %context
  // CHECK: %[[stream:.*]] = tfrt_gpu.stream.create %[[context]]
  %stream = tfrt_gpu.stream.create %context

  %ch0 = tfrt.new.chain
  // CHECK: %[[size:.*]] = tfrt.constant.i64 1024
  %size = tfrt.constant.i64 1024
  // CHECK: %[[buffer:.*]] = tfrt_gpu.mem.allocate %[[allocator]], %[[stream]], %[[size]], %{{.*}}
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0

  // CHECK: %[[blas:.*]] = tfrt_gpu.blas.create %[[stream]]
  %blas = tfrt_gpu.blas.create %stream
  // CHECK: %[[width:.*]] = tfrt.constant.i32 2
  %width = tfrt.constant.i32 2
  // CHECK: %[[stride:.*]] = tfrt.constant.i32 1
  %stride = tfrt.constant.i32 1
  // CHECK: %[[alpha:.*]] = tfrt.constant.f32 1.0
  %alpha = tfrt.constant.f32 1.0
  // CHECK: tfrt_gpu.blas.axpy %[[blas]], %[[width]], %[[alpha]], CUDA_R_32F,
  // CHECK-SAME: %[[buffer]], CUDA_R_32F, %[[stride]], %[[buffer]], CUDA_R_32F,
  // CHECK-SAME: %[[stride]], CUDA_R_32F, %{{.*}}
  %ch1 = tfrt_gpu.blas.axpy %blas, %width, %alpha, CUDA_R_32F,
    %buffer, CUDA_R_32F, %stride, %buffer, CUDA_R_32F, %stride,
    CUDA_R_32F, %ch0

  // CHECK: %[[algo:.*]] = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_ALGO0
  %algo = tfrt_gpu.blas.gemm.algo CUBLAS_GEMM_ALGO0
  // CHECK: tfrt_gpu.blas.gemm %[[blas]], CUBLAS_OP_N, CUBLAS_OP_N,
  // CHECK-SAME: %[[stride]], %[[stride]], %[[stride]], %[[alpha]], %[[buffer]],
  // CHECK-SAME: CUDA_R_32F, %[[stride]], %[[buffer]], CUDA_R_32F, %[[stride]],
  // CHECK-SAME: %[[alpha]], %[[buffer]], CUDA_R_32F, %[[stride]],
  // CHECK-SAME: CUBLAS_COMPUTE_32F, %[[algo]], %{{.*}}
  %ch2 = tfrt_gpu.blas.gemm %blas, CUBLAS_OP_N, CUBLAS_OP_N,
    %stride, %stride, %stride, %alpha, %buffer,
    CUDA_R_32F, %stride, %buffer, CUDA_R_32F, %stride,
    %alpha, %buffer, CUDA_R_32F, %stride, CUBLAS_COMPUTE_32F,
    %algo, %ch1
  // CHECK: tfrt_gpu.blas.gemm.batch %[[blas]], CUBLAS_OP_N, CUBLAS_OP_N,
  // CHECK-SAME: %[[stride]], %[[stride]], %[[stride]], %[[alpha]], %[[buffer]],
  // CHECK-SAME: CUDA_R_32F, %[[stride]], %[[size]], %[[buffer]], CUDA_R_32F, %[[stride]],
  // CHECK-SAME: %[[size]], %[[alpha]], %[[buffer]], CUDA_R_32F, %[[stride]], %[[size]], %[[stride]],
  // CHECK-SAME: CUBLAS_COMPUTE_32F, %[[algo]], %{{.*}}
  %ch3 = tfrt_gpu.blas.gemm.batch %blas, CUBLAS_OP_N, CUBLAS_OP_N,
    %stride, %stride, %stride, %alpha, %buffer,
    CUDA_R_32F, %stride, %size, %buffer, CUDA_R_32F, %stride,
    %size, %alpha, %buffer, CUDA_R_32F, %stride, %size, %stride,
    CUBLAS_COMPUTE_32F, %algo, %ch2
  // CHECK: tfrt_gpu.blas.trsm.batch %[[blas]], CUBLAS_SIDE_LEFT,
  // CHECK-SAME: CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
  // CHECK-SAME: %[[stride]], %[[stride]], CUDA_R_32F, %[[alpha]], %[[buffer]],
  // CHECK-SAME: %[[stride]], %[[buffer]], %[[stride]], %[[stride]], %{{.*}}
  %ch4 = tfrt_gpu.blas.trsm.batch %blas, CUBLAS_SIDE_LEFT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, %stride, %stride,
    CUDA_R_32F, %alpha, %buffer, %stride, %buffer, %stride, %stride, %ch3

  tfrt.return
}

func @dnn_ops() {
  // CHECK: %[[ordinal:.*]] = tfrt.constant.i32 0
  %ordinal = tfrt.constant.i32 0
  // CHECK: %[[device:.*]] = tfrt_gpu.device.get CUDA, %[[ordinal]]
  %device = tfrt_gpu.device.get CUDA, %ordinal
  // CHECK: %[[context:.*]] = tfrt_gpu.context.create %[[device]]
  %context = tfrt_gpu.context.create %device
  // CHECK: %[[stream:.*]] = tfrt_gpu.stream.create %[[context]]
  %stream = tfrt_gpu.stream.create %context

  // CHECK: %[[cudnn:.*]] = tfrt_gpu.dnn.create %[[stream]]
  %cudnn = tfrt_gpu.dnn.create %stream
  // TODO(csigg): cover other dnn ops.

  tfrt.return
}

func @solver_ops() {
  // CHECK: %[[ordinal:.*]] = tfrt.constant.i32 0
  %ordinal = tfrt.constant.i32 0
  // CHECK: %[[device:.*]] = tfrt_gpu.device.get CUDA, %[[ordinal]]
  %device = tfrt_gpu.device.get CUDA, %ordinal
  // CHECK: %[[context:.*]] = tfrt_gpu.context.create %[[device]]
  %context = tfrt_gpu.context.create %device
  // CHECK: %[[allocator:.*]] = tfrt_gpu.allocator.create %[[context]]
  %allocator = tfrt_gpu.allocator.create %context
  // CHECK: %[[stream:.*]] = tfrt_gpu.stream.create %[[context]]
  %stream = tfrt_gpu.stream.create %context

  %ch0 = tfrt.new.chain
  // CHECK: %[[size:.*]] = tfrt.constant.i64 1024
  %size = tfrt.constant.i64 1024
  // CHECK: %[[buffer:.*]] = tfrt_gpu.mem.allocate %[[allocator]], %[[stream]], %[[size]], %{{.*}}
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0

  // CHECK: %[[n:.*]] = tfrt.constant.i32 1
  %n = tfrt.constant.i32 1

  // CHECK: %[[solver:.*]] = tfrt_gpu.solver.create %[[stream]]
  %solver = tfrt_gpu.solver.create %stream
  // CHECK: %[[workspace_size:.*]] = tfrt_gpu.solver.potrf.buffer_size %[[solver]],
  // CHECK-SAME: CUBLAS_FILL_MODE_LOWER, %[[n]], CUDA_R_32F, %[[n]], %{{.*}}
  %workspace_size = tfrt_gpu.solver.potrf.buffer_size %solver,
    CUBLAS_FILL_MODE_LOWER, %n, CUDA_R_32F, %n, %ch0
  // CHECK: tfrt_gpu.solver.potrf %[[solver]], CUBLAS_FILL_MODE_LOWER, %[[n]],
  // CHECK-SAME: CUDA_R_32F, %[[buffer]], %[[n]], %[[buffer]], %[[buffer]], %{{.*}}
  %ch1 = tfrt_gpu.solver.potrf %solver, CUBLAS_FILL_MODE_LOWER, %n, CUDA_R_32F,
    %buffer, %n, %buffer, %buffer, %ch0
  // CHECK: tfrt_gpu.solver.potrf.batch %[[solver]], CUBLAS_FILL_MODE_LOWER,
  // CHECK-SAME: %[[n]], CUDA_R_32F, %[[buffer]], %[[n]], %[[buffer]], %[[n]],
  // CHECK-SAME: %{{.*}}
  %ch2 = tfrt_gpu.solver.potrf.batch %solver, CUBLAS_FILL_MODE_LOWER, %n,
    CUDA_R_32F, %buffer, %n, %buffer, %n, %ch1

  tfrt.return
}

func @ccl_ops() {
  // CHECK: %[[ordinal:.*]] = tfrt.constant.i32 0
  %ordinal = tfrt.constant.i32 0
  // CHECK: %[[device:.*]] = tfrt_gpu.device.get CUDA, %[[ordinal]]
  %device = tfrt_gpu.device.get CUDA, %ordinal
  // CHECK: %[[context:.*]] = tfrt_gpu.context.create %[[device]]
  %context = tfrt_gpu.context.create %device
  // CHECK: %[[allocator:.*]] = tfrt_gpu.allocator.create %[[context]]
  %allocator = tfrt_gpu.allocator.create %context
  // CHECK: %[[stream:.*]] = tfrt_gpu.stream.create %[[context]]
  %stream = tfrt_gpu.stream.create %context

  %ch0 = tfrt.new.chain
  // CHECK: %[[size:.*]] = tfrt.constant.i64 1024
  %size = tfrt.constant.i64 1024
  // CHECK: %[[buffer:.*]] = tfrt_gpu.mem.allocate %[[allocator]], %[[stream]], %[[size]], %{{.*}}
  %buffer = tfrt_gpu.mem.allocate %allocator, %stream, %size, %ch0

  // CHECK: %[[rank:.*]] = tfrt.constant.i32 0
  %rank = tfrt.constant.i32 0
  // CHECK: %[[count:.*]] = tfrt.constant.i32 1
  %count = tfrt.constant.i32 1
  // CHECK: %[[id:.*]] = tfrt_gpu.ccl.unique_id CUDA
  %id = tfrt_gpu.ccl.unique_id CUDA
  // CHECK: %[[ccl:.*]] = tfrt_gpu.ccl.create %[[context]], %[[rank]], %[[count]], %[[id]]
  %ccl = tfrt_gpu.ccl.create %context, %rank, %count, %id
  // CHECK: tfrt_gpu.ccl.all_gather %[[ccl]], %[[buffer]], %[[buffer]], ncclFloat32, %{{.*}}
  %ch1 = tfrt_gpu.ccl.all_gather %ccl, %buffer, %buffer, ncclFloat32, %ch0
  // CHECK: tfrt_gpu.ccl.all_reduce %[[ccl]], %[[buffer]], %[[buffer]], ncclFloat32, ncclSum, %{{.*}}
  %ch2 = tfrt_gpu.ccl.all_reduce %ccl, %buffer, %buffer, ncclFloat32, ncclSum, %ch1
  // CHECK: tfrt_gpu.ccl.reduce_scatter %[[ccl]], %[[buffer]], %[[buffer]], ncclFloat32, ncclSum, %{{.*}}
  %ch3 = tfrt_gpu.ccl.reduce_scatter %ccl, %buffer, %buffer, ncclFloat32, ncclSum, %ch2
  // CHECK: tfrt_gpu.ccl.execute %[[stream]], %[[ccl]], %{{.*}}
  %ch4 = tfrt_gpu.ccl.execute %stream, %ccl, %ch3

  tfrt.return
}
