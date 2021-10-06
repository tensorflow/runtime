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

def GPU_BlasDataTypeAttr : GPU_WrapperAttr<"BlasDataType">;
def GPU_BlasDiagTypeAttr : GPU_WrapperAttr<"BlasDiagType">;
def GPU_BlasComputeTypeAttr : GPU_WrapperAttr<"BlasComputeType">;
def GPU_BlasOperationAttr : GPU_WrapperAttr<"BlasOperation">;
def GPU_BlasGemmAlgoAttr : GPU_WrapperAttr<"BlasGemmAlgo">;
def GPU_BlasFillModeAttr : GPU_WrapperAttr<"BlasFillMode">;
def GPU_BlasSideModeAttr : GPU_WrapperAttr<"BlasSideMode">;

def GPU_BlasCreateOp : GPU_Op<"blas.create"> {
  let description = [{
    tfrt_gpu.blas.create creates a handle to the BLAS library context

    This will be explicitly passed to every subsequent BLAS library call.
  }];
  let arguments = (ins GPU_StreamType);
  let results = (outs GPU_BlasHandleType);
}

def GPU_BlasSaxpyOp : GPU_Op<"blas.axpy"> {
  let description = [{
    tfrt_gpu.blas.axpy call BLAS axpy function.
  }];
  let arguments = (ins GPU_BlasHandleType:$handle, I32:$n, AnyType:$alpha,
                   GPU_BlasDataTypeAttr:$typeAlpha, GPU_BufferType:$x,
                   GPU_BlasDataTypeAttr:$typeX, I32:$strideX, GPU_BufferType:$y,
                   GPU_BlasDataTypeAttr:$typeY, I32:$strideY,
                   GPU_BlasDataTypeAttr:$executionType, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $n`,` $alpha`,` custom<Enum>($typeAlpha, type($alpha))`,`
    $x`,` custom<Enum>($typeX)`,` $strideX`,` $y`,` custom<Enum>($typeY)`,`
    $strideY`,` custom<Enum>($executionType)`,` $chain attr-dict
  }];
  let verifier = [{ return tfrt::gpu::VerifyBlasSaxpyOp(*this); }];
}

def GPU_BlasGemmAlgoOp : GPU_Op<"blas.gemm.algo", [NoSideEffect]> {
  let arguments = (ins GPU_BlasGemmAlgoAttr:$algo);
  let results = (outs GPU_BlasGemmAlgoType);
  let assemblyFormat = "custom<Enum>($algo) attr-dict";
}

def GPU_BlasGemmOp : GPU_Op<"blas.gemm"> {
  let description = [{
    tfrt_gpu.blas.gemm This kernel is an extension of blas.gemm.<t> kernels
                       that allows the user to individually specify the data
                       types for each of the A, B and C matrices, the precision
                       of computation and the GEMM algorithm to be run.
  }];
  let arguments = (ins GPU_BlasHandleType:$handle, GPU_BlasOperationAttr:$transA,
    GPU_BlasOperationAttr:$transB, I32:$m, I32:$n, I32:$k, AnyType:$alpha,
    GPU_BufferType:$A, GPU_BlasDataTypeAttr:$typeA, I32:$heightA, GPU_BufferType:$B,
    GPU_BlasDataTypeAttr:$typeB, I32:$heightB, AnyType:$beta, GPU_BufferType:$C,
    GPU_BlasDataTypeAttr:$typeC, I32:$heightC, GPU_BlasComputeTypeAttr:$computeType,
    GPU_BlasGemmAlgoType:$algo, TFRT_ChainType:$chain);

  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` custom<Enum>($transA)`,` custom<Enum>($transB)`,`
    $m`,` $n`,` $k`,` $alpha`,` $A`,` custom<Enum>($typeA)`,` $heightA`,` $B`,`
    custom<Enum>($typeB)`,` $heightB`,` $beta`,` $C`,` custom<Enum>($typeC)`,`
    $heightC`,` custom<Enum>($computeType, type($alpha), type($beta))`,`
    $algo`,` $chain attr-dict
  }];
  let verifier = [{ return tfrt::gpu::VerifyBlasGemmOp(*this); }];
}

def GPU_BlasGemmBatchExOp : GPU_Op<"blas.gemm.batch"> {
  let description = [{
    tfrt_gpu.blas.gemm.batch This kernel is an extension of blas.gemm that
                           performs the matrix-matrix multiplication of a batch
                           of matrices and allows the individual specification
                           strides between consecutive elements.
  }];
  let arguments = (ins GPU_BlasHandleType:$handle, GPU_BlasOperationAttr:$transA,
    GPU_BlasOperationAttr:$transB, I32:$m, I32:$n, I32:$k, AnyType:$alpha,
    GPU_BufferType:$A, GPU_BlasDataTypeAttr:$typeA, I32:$heightA, I64:$strideA,
    GPU_BufferType:$B, GPU_BlasDataTypeAttr:$typeB, I32:$heightB, I64:$strideB,
    AnyType:$beta,  GPU_BufferType:$C, GPU_BlasDataTypeAttr:$typeC, I32:$heightC,
    I64:$strideC, I32:$batchCount, GPU_BlasComputeTypeAttr:$computeType,
    GPU_BlasGemmAlgoType:$algo, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` custom<Enum>($transA)`,` custom<Enum>($transB)`,`
    $m`,` $n`,` $k`,` $alpha`,` $A`,` custom<Enum>($typeA)`,` $heightA
    `,` $strideA`,` $B`,` custom<Enum>($typeB)`,` $heightB`,` $strideB`,` $beta`,`
    $C`,` custom<Enum>($typeC)`,` $heightC`,` $strideC`,` $batchCount`,`
    custom<Enum>($computeType, type($alpha), type($beta))`,`
    $algo`,` $chain attr-dict
  }];
  let verifier = [{ return tfrt::gpu::VerifyBlasGemmOp(*this); }];
}

def GPU_BlasTrsmBatchOp : GPU_Op<"blas.trsm.batch"> {
  let description = [{
    tfrt_gpu.blas.trsm.batch solves an array of triangular linear systems with
    multiple right-hand-sides using the specified batch size.
  }];
  let arguments = (ins GPU_BlasHandleType:$handle, GPU_BlasSideModeAttr:$sideMode,
                   GPU_BlasFillModeAttr:$fillMode, GPU_BlasOperationAttr:$trans,
                   GPU_BlasDiagTypeAttr:$diagType, I32:$m, I32:$n,
                   GPU_BlasDataTypeAttr:$dataType, AnyType:$alpha,
                   GPU_BufferType:$A, I32:$heightA, GPU_BufferType:$B,
                   I32:$heightB, I32:$batchCount, TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` custom<Enum>($sideMode)`,` custom<Enum>($fillMode)`,`
    custom<Enum>($trans)`,` custom<Enum>($diagType)`,` $m`,` $n`,`
    custom<Enum>($dataType, type($alpha))`,` $alpha`,` $A`,` $heightA`,` $B`,`
    $heightB`,` $batchCount`,` $chain attr-dict
  }];
}

#endif  // GPU_OPS
