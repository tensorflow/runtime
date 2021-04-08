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

//===- cuda_ops.td ---------------------------------------------------------===//
//
// CUDA based operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_BLAS_OPS
#else
#define CUDA_BLAS_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td"

def AnyCublasDataType : AnyTypeOf<[AnyFloat, AnyComplex, AnySignedInteger,
                                   AnyUnsignedInteger]>;

def BlasCreateOp : CUDA_Op<"blas.create"> {
  let summary = "tfrt_cuda blas.create operation";
  let description = [{
    tfrt_cuda.blas.create creates a handle to the BLAS library context

    This will be explicitly passed to every subsequent BLAS library call.
  }];
  let arguments = (ins ContextType);
  let results = (outs CublasHandleType);
}

def BlasSetStreamOp : CUDA_Op<"blas.set_stream"> {
  let summary = "tfrt_cuda blas.set_stream operation";
  let description = [{
    tfrt_cuda.blas.set_stream sets the BLAS library stream.

    All subsequent BLAS calls will execute on the set stream (until
    blas.set_stream is called with a different stream).
  }];
  let arguments = (ins CublasHandleType, StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def BlasSaxpyOp : CUDA_Op<"blas.axpy.f32"> {
  let summary = "tfrt_cuda blas.axpy.f32 operation";
  let description = [{
    tfrt_cuda.blas.axpy.f32 call BLAS saxpy function.
  }];
  let arguments = (ins ContextType, CublasHandleType, I32:$n, F32:$alpha,
                   BufferType:$x, I32:$incx, BufferType:$y, I32:$incy,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def BlasSgemmOp : CUDA_Op<"blas.gemm.f32"> {
  let summary = "tfrt_cuda blas.gemm.f32 operation";
  let description = [{
    tfrt_cuda.blas.gemm.f32 calls BLAS sgemm function.
  }];
  let arguments = (ins ContextType, CublasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k, F32:$alpha, BufferType:$A, I32:$lda,
                   BufferType:$B, I32:$ldb, F32:$beta, BufferType:$C, I32:$ldc,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def BlasGemmExOp : CUDA_Op<"blas.gemm.ex"> {
  let summary = "tfrt_cuda blas.gemm.ex operation";
  let description = [{
    tfrt_cuda.blas.gemm.ex This kernel is an extension of blas.gemm.<t> kernels
                           that allows the user to individually specify the
                           data types for each of the A, B and C matrices,
                           the precision of computation and the GEMM
                           algorithm to be run.
  }];
  let arguments = (ins ContextType, CublasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k,
                   F32:$alpha,
                   BufferType:$A, I32:$Atype, I32:$lda,
                   BufferType:$B, I32:$Btype, I32:$ldb,
                   F32:$beta,
                   BufferType:$C, I32:$Ctype, I32:$ldc,
                   I32:$computeType, I32:$algo, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def BlasGemmStridedBatchedExOp : CUDA_Op<"blas.gemm.strided.batched.ex"> {
  let summary = "tfrt_cuda blas.gemm.batched.ex operation";
  let description = [{
    tfrt_cuda.blas.gemm.ex This kernel is an extension of blas.gemm.ex that
                           performs the matrix-matrix multiplication of a batch
                           of matrices and allows the individual specification
                           of the data types for each of the A, B and C matrix
                           arrays, the precision of computation and the GEMM
                           algorithm to be run. Address offsets strideA,
                           strideB and strideC determine the locations of input
                           and output matrices in future instances
  }];
  let arguments = (ins ContextType, CublasHandleType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k,
                   F32:$alpha,
                   BufferType:$A, I32:$Atype, I32:$lda, I64:$strideA,
                   BufferType:$B, I32:$Btype, I32:$ldb, I64:$strideB,
                   F32:$beta,
                   BufferType:$C, I32:$Ctype, I32:$ldc, I64:$strideC,
                   I32:$batch_count,
                   I32:$computeType, I32:$algo, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

// TODO(b/175169391): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def BlasSyncSgemmOp : CUDA_Op<"blas.sync.gemm.f32"> {
  let summary = "tfrt_cuda blas.sync.gemm.f32 operation";
  let description = [{
    tfrt_cuda.blas.sync.gemm.f32 calls BLAS sgemm function.
  }];
  let arguments = (ins ContextType, CublasHandleType, StreamType,
                   BoolAttr:$transa, BoolAttr:$transb,
                   I32:$m, I32:$n, I32:$k, F32:$alpha, BufferType:$A, I32:$lda,
                   BufferType:$B, I32:$ldb, F32:$beta, BufferType:$C, I32:$ldc);
}

// TODO(b/175169391): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def BlasSyncGemmExOp : Op<CUDA_Dialect, "blas.sync.gemm_ex"> {
  let summary = "tfrt_cuda blas.sync.gemm_ex operation";
  let description = [{
    tfrt_cuda.blas.sync.gemm_ex calls BLAS GemmEx function.
  }];
  // Note: keep the attributes alphabetically sorted, because BEF encodes them
  // in the getAttrs() order, which is sorted only if getAttrDictionary() is
  // called before. See b/184656568.
  // TODO(csigg): Investigate whether MLIR would want a stable attribute order.
  let arguments = (ins ContextType, CublasHandleType, StreamType,
                   I32:$m, I32:$n, I32:$k,
                   AnyCublasDataType:$alpha, BufferType:$A, I32:$Atype,
                   I32:$lda, BufferType:$B, I32:$Btype, I32:$ldb,
                   AnyCublasDataType:$beta, BufferType:$C, I32:$Ctype, I32:$ldc,
                   I32:$algorithm, I32Attr:$computeType,
                   BoolAttr:$transa, BoolAttr:$transb);
}

#endif  // CUDA_OPS
