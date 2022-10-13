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

#ifdef TFRTGPU_BLAS_OPS
#else
#define TFRTGPU_BLAS_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TFRTGPU_BlasHandleType : TFRTGPU_Type<"BlasHandle"> {
  let mnemonic = "blas.handle";
}
def TFRTGPU_BlasGemmAlgoType : TFRTGPU_Type<"BlasGemmAlgo"> {
  let mnemonic = "blas.gemm.algo";
}

def TFRTGPU_BlasDataTypeAttr : TFRTGPU_WrapperAttr<"BlasDataType">;
def TFRTGPU_BlasDiagTypeAttr : TFRTGPU_WrapperAttr<"BlasDiagType">;
def TFRTGPU_BlasComputeTypeAttr : TFRTGPU_WrapperAttr<"BlasComputeType">;
def TFRTGPU_BlasOperationAttr : TFRTGPU_WrapperAttr<"BlasOperation">;
def TFRTGPU_BlasGemmAlgoAttr : TFRTGPU_WrapperAttr<"BlasGemmAlgo">;
def TFRTGPU_BlasFillModeAttr : TFRTGPU_WrapperAttr<"BlasFillMode">;
def TFRTGPU_BlasSideModeAttr : TFRTGPU_WrapperAttr<"BlasSideMode">;

def TFRTGPU_BlasCreateOp : TFRTGPU_Op<"blas.create"> {
  let description = [{
    tfrt_gpu.blas.create creates a handle to the BLAS library context.

    This will be explicitly passed to every subsequent BLAS library call.
  }];
  let arguments = (ins TFRTGPU_ContextType);
  let results = (outs TFRTGPU_BlasHandleType);
}

def TFRTGPU_BlasSaxpyOp : TFRTGPU_Op<"blas.axpy"> {
  let description = [{
    tfrt_gpu.blas.axpy call BLAS axpy function.
  }];
  let arguments = (ins TFRTGPU_BlasHandleType:$handle, TFRTGPU_StreamType:$stream, I32:$n, AnyType:$alpha,
                   TFRTGPU_BlasDataTypeAttr:$typeAlpha, TFRTGPU_BufferType:$x,
                   TFRTGPU_BlasDataTypeAttr:$typeX, I32:$strideX, TFRTGPU_BufferType:$y,
                   TFRTGPU_BlasDataTypeAttr:$typeY, I32:$strideY,
                   TFRTGPU_BlasDataTypeAttr:$executionType, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` $n`,` $alpha`,` custom<Enum>($typeAlpha, type($alpha))`,`
    $x`,` custom<Enum>($typeX)`,` $strideX`,` $y`,` custom<Enum>($typeY)`,`
    $strideY`,` custom<Enum>($executionType)`,` $chain attr-dict
  }];
  let hasVerifier = 1;
}

def TFRTGPU_BlasGemmAlgoOp : TFRTGPU_Op<"blas.gemm.algo", [Pure]> {
  let arguments = (ins TFRTGPU_BlasGemmAlgoAttr:$algo);
  let results = (outs TFRTGPU_BlasGemmAlgoType);
  let assemblyFormat = "custom<Enum>($algo) attr-dict";
}

def TFRTGPU_BlasGemmOp : TFRTGPU_Op<"blas.gemm"> {
  let description = [{
    tfrt_gpu.blas.gemm This kernel is an extension of blas.gemm.<t> kernels
                       that allows the user to individually specify the data
                       types for each of the A, B and C matrices, the precision
                       of computation and the GEMM algorithm to be run.
  }];
  let arguments = (ins TFRTGPU_BlasHandleType:$handle, TFRTGPU_StreamType:$stream, TFRTGPU_BlasOperationAttr:$transA,
    TFRTGPU_BlasOperationAttr:$transB, I32:$m, I32:$n, I32:$k, AnyType:$alpha,
    TFRTGPU_BufferType:$A, TFRTGPU_BlasDataTypeAttr:$typeA, I32:$heightA, TFRTGPU_BufferType:$B,
    TFRTGPU_BlasDataTypeAttr:$typeB, I32:$heightB, AnyType:$beta, TFRTGPU_BufferType:$C,
    TFRTGPU_BlasDataTypeAttr:$typeC, I32:$heightC, TFRTGPU_BlasComputeTypeAttr:$computeType,
    TFRTGPU_BlasGemmAlgoType:$algo, TFRT_ChainType:$chain);

  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($transA)`,` custom<Enum>($transB)`,`
    $m`,` $n`,` $k`,` $alpha`,` $A`,` custom<Enum>($typeA)`,` $heightA`,` $B`,`
    custom<Enum>($typeB)`,` $heightB`,` $beta`,` $C`,` custom<Enum>($typeC)`,`
    $heightC`,` custom<Enum>($computeType, type($alpha), type($beta))`,`
    $algo`,` $chain attr-dict
  }];
  let hasVerifier = 1;
}

def TFRTGPU_BlasGemmBatchExOp : TFRTGPU_Op<"blas.gemm.batch"> {
  let description = [{
    tfrt_gpu.blas.gemm.batch This kernel is an extension of blas.gemm that
                           performs the matrix-matrix multiplication of a batch
                           of matrices and allows the individual specification
                           strides between consecutive elements.
  }];
  let arguments = (ins TFRTGPU_BlasHandleType:$handle, TFRTGPU_StreamType:$stream, TFRTGPU_BlasOperationAttr:$transA,
    TFRTGPU_BlasOperationAttr:$transB, I32:$m, I32:$n, I32:$k, AnyType:$alpha,
    TFRTGPU_BufferType:$A, TFRTGPU_BlasDataTypeAttr:$typeA, I32:$heightA, I64:$strideA,
    TFRTGPU_BufferType:$B, TFRTGPU_BlasDataTypeAttr:$typeB, I32:$heightB, I64:$strideB,
    AnyType:$beta,  TFRTGPU_BufferType:$C, TFRTGPU_BlasDataTypeAttr:$typeC, I32:$heightC,
    I64:$strideC, I32:$batchCount, TFRTGPU_BlasComputeTypeAttr:$computeType,
    TFRTGPU_BlasGemmAlgoType:$algo, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($transA)`,` custom<Enum>($transB)`,`
    $m`,` $n`,` $k`,` $alpha`,` $A`,` custom<Enum>($typeA)`,` $heightA
    `,` $strideA`,` $B`,` custom<Enum>($typeB)`,` $heightB`,` $strideB`,` $beta`,`
    $C`,` custom<Enum>($typeC)`,` $heightC`,` $strideC`,` $batchCount`,`
    custom<Enum>($computeType, type($alpha), type($beta))`,`
    $algo`,` $chain attr-dict
  }];
  let hasVerifier = 1;
}

def TFRTGPU_BlasScalOp : TFRTGPU_Op<"blas.scal"> {
  let description = [{
    tfrt_gpu.blas.scal call BLAS scal function.
  }];
  let arguments = (ins TFRTGPU_BlasHandleType:$handle, TFRTGPU_StreamType:$stream, I32:$n,
                   AnyType:$alpha, TFRTGPU_BlasDataTypeAttr:$typeAlpha,
                   TFRTGPU_BufferType:$x, TFRTGPU_BlasDataTypeAttr:$typeX, I32:$strideX,
                   TFRTGPU_BlasDataTypeAttr:$executionType, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` $n`,` $alpha`,` custom<Enum>($typeAlpha, type($alpha))`,`
    $x`,` custom<Enum>($typeX)`,` $strideX`,`
    custom<Enum>($executionType)`,` $chain attr-dict
  }];
  let hasVerifier = 1;
}

def TFRTGPU_BlasTrsmBatchOp : TFRTGPU_Op<"blas.trsm.batch"> {
  let description = [{
    tfrt_gpu.blas.trsm.batch solves an array of triangular linear systems with
    multiple right-hand-sides using the specified batch size.
  }];
  let arguments = (ins TFRTGPU_BlasHandleType:$handle, TFRTGPU_StreamType:$stream, TFRTGPU_BlasSideModeAttr:$sideMode,
                   TFRTGPU_BlasFillModeAttr:$fillMode, TFRTGPU_BlasOperationAttr:$trans,
                   TFRTGPU_BlasDiagTypeAttr:$diagType, I32:$m, I32:$n,
                   TFRTGPU_BlasDataTypeAttr:$dataType, AnyType:$alpha,
                   TFRTGPU_BufferType:$A, I32:$heightA, TFRTGPU_BufferType:$B,
                   I32:$heightB, I32:$batchCount, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($sideMode)`,` custom<Enum>($fillMode)`,`
    custom<Enum>($trans)`,` custom<Enum>($diagType)`,` $m`,` $n`,`
    custom<Enum>($dataType, type($alpha))`,` $alpha`,` $A`,` $heightA`,` $B`,`
    $heightB`,` $batchCount`,` $chain attr-dict
  }];
}

#endif  // TFRTGPU_OPS
