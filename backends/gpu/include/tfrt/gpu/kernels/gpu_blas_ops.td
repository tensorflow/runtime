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
include "mlir/Interfaces/SideEffectInterfaces.td"

def GPU_BlasHandleType : GPU_Type<"BlasHandle"> {
  let mnemonic = "blas.handle";
}
def GPU_BlasGemmAlgoType : GPU_Type<"BlasGemmAlgo"> {
  let mnemonic = "blas.gemm.algo";
}
def GPU_AnyBlasDataType : AnyTypeOf<[AnyFloat, AnyComplex, AnySignedInteger,
                                   AnyUnsignedInteger]>;

def GPU_BlasDataTypeAttr : GPU_WrapperAttr<"BlasDataType">;
def GPU_BlasOperationAttr : GPU_WrapperAttr<"BlasOperation">;
def GPU_BlasGemmAlgoAttr : GPU_WrapperAttr<"BlasGemmAlgo">;

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

def GPU_BlasGemmAlgoOp : GPU_Op<"blas.gemm.algo", [NoSideEffect]> {
  let arguments = (ins GPU_BlasGemmAlgoAttr:$algo);
  let results = (outs GPU_BlasGemmAlgoType);
  let builders = [
    OpBuilder<(ins "::tfrt::gpu::wrapper::BlasGemmAlgo":$algo),
    [{
      auto context = $_builder.getContext();
      auto type = BlasGemmAlgoType::get(context);
      auto attr = BlasGemmAlgoAttr::get(context, algo);
      return build($_builder, $_state, type, attr);
    }]>];
  let assemblyFormat = "custom<Enum>($algo) attr-dict";
}

def GPU_BlasGemmOp : GPU_Op<"blas.gemm"> {
  let description = [{
    tfrt_gpu.blas.gemm This kernel is an extension of blas.gemm.<t> kernels
                       that allows the user to individually specify the data
                       types for each of the A, B and C matrices, the precision
                       of computation and the GEMM algorithm to be run.
  }];
  let arguments = (ins GPU_BlasHandleType:$handle, GPU_BlasOperationAttr:$transa,
    GPU_BlasOperationAttr:$transb, I32:$m, I32:$n, I32:$k, F32:$alpha,
    GPU_BufferType:$A, GPU_BlasDataTypeAttr:$Atype, I32:$lda, GPU_BufferType:$B,
    GPU_BlasDataTypeAttr:$Btype, I32:$ldb, F32:$beta, GPU_BufferType:$C,
    GPU_BlasDataTypeAttr:$Ctype, I32:$ldc, GPU_BlasDataTypeAttr:$computeType,
    GPU_BlasGemmAlgoType:$algo, TFRT_ChainType:$chain);

  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` custom<Enum>($transa)`,` custom<Enum>($transb)`,`
    $m`,` $n`,` $k`,` $alpha`,` $A`,` custom<Enum>($Atype)`,` $lda`,` $B`,`
    custom<Enum>($Btype)`,` $ldb`,` $beta`,` $C`,` custom<Enum>($Ctype)`,`
    $ldc`,` custom<Enum>($computeType)`,` $algo`,` $chain
    attr-dict
  }];
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
