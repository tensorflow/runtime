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

//===- gpu_ops.td ---------------------------------------------------------===//
//
// CUDA based operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_BLAS_OPS
#else
#define GPU_BLAS_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

def GPU_BlasHandleType : GPU_Type<"blas_handle">;
def GPU_AnyBlasDataType : AnyTypeOf<[AnyFloat, AnyComplex, AnySignedInteger,
                                   AnyUnsignedInteger]>;

def GPU_BlasCreateOp : GPU_Op<"blas.create"> {
  let description = [{
    tfrt_gpu.blas.create creates a handle to the BLAS library context

    This will be explicitly passed to every subsequent BLAS library call.
  }];
  let arguments = (ins GPU_StreamType, TFRT_ChainType);
  let results = (outs GPU_BlasHandleType);
}

def GPU_BlasSaxpyOp : GPU_Op<"blas.axpy.f32"> {
  let description = [{
    tfrt_gpu.blas.axpy.f32 call BLAS saxpy function.
  }];
  let arguments = (ins GPU_BlasHandleType, I32:$n, F32:$alpha,
                   GPU_BufferType:$x, I32:$incx, GPU_BufferType:$y, I32:$incy,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_BlasSgemmOp : GPU_Op<"blas.gemm.f32"> {
  let description = [{
    tfrt_gpu.blas.gemm.f32 calls BLAS sgemm function.
  }];
  let arguments = (ins GPU_BlasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k, F32:$alpha, GPU_BufferType:$A, I32:$lda,
                   GPU_BufferType:$B, I32:$ldb, F32:$beta, GPU_BufferType:$C, I32:$ldc,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_BlasGemmExOp : GPU_Op<"blas.gemm.ex"> {
  let description = [{
    tfrt_gpu.blas.gemm.ex This kernel is an extension of blas.gemm.<t> kernels
                           that allows the user to individually specify the
                           data types for each of the A, B and C matrices,
                           the precision of computation and the GEMM
                           algorithm to be run.
  }];
  let arguments = (ins GPU_BlasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k,
                   F32:$alpha,
                   GPU_BufferType:$A, I32:$Atype, I32:$lda,
                   GPU_BufferType:$B, I32:$Btype, I32:$ldb,
                   F32:$beta,
                   GPU_BufferType:$C, I32:$Ctype, I32:$ldc,
                   I32:$computeType, I32:$algo, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def GPU_BlasGemmStridedBatchedExOp : GPU_Op<"blas.gemm.strided.batched.ex"> {
  let description = [{
    tfrt_gpu.blas.gemm.ex This kernel is an extension of blas.gemm.ex that
                           performs the matrix-matrix multiplication of a batch
                           of matrices and allows the individual specification
                           of the data types for each of the A, B and C matrix
                           arrays, the precision of computation and the GEMM
                           algorithm to be run. Address offsets strideA,
                           strideB and strideC determine the locations of input
                           and output matrices in future instances
  }];
  let arguments = (ins GPU_BlasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k,
                   F32:$alpha,
                   GPU_BufferType:$A, I32:$Atype, I32:$lda, I64:$strideA,
                   GPU_BufferType:$B, I32:$Btype, I32:$ldb, I64:$strideB,
                   F32:$beta,
                   GPU_BufferType:$C, I32:$Ctype, I32:$ldc, I64:$strideC,
                   I32:$batch_count,
                   I32:$computeType, I32:$algo, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

// TODO(b/175169391): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_BlasSyncSgemmOp : GPU_Op<"blas.sync.gemm.f32"> {
  let description = [{
    tfrt_gpu.blas.sync.gemm.f32 calls BLAS sgemm function.
  }];
  let arguments = (ins GPU_BlasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k, F32:$alpha, GPU_BufferType:$A, I32:$lda,
                   GPU_BufferType:$B, I32:$ldb, F32:$beta, GPU_BufferType:$C, I32:$ldc,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

// TODO(b/175169391): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_BlasSyncGemmExOp : Op<GPU_Dialect, "blas.sync.gemm_ex"> {
  let description = [{
    tfrt_gpu.blas.sync.gemm_ex calls BLAS GemmEx function.
  }];
  let arguments = (ins GPU_BlasHandleType,
                   I32:$m, I32:$n, I32:$k, GPU_AnyBlasDataType:$alpha,
                   GPU_BufferType:$A, I32:$Atype, I32:$lda, GPU_BufferType:$B,
                   I32:$Btype, I32:$ldb, GPU_AnyBlasDataType:$beta,
                   GPU_BufferType:$C, I32:$Ctype, I32:$ldc, I32:$algorithm,
                   TFRT_ChainType, I32Attr:$computeType, BoolAttr:$transa,
                   BoolAttr:$transb);
  let results = (outs TFRT_ChainType);
}

#endif  // GPU_OPS
