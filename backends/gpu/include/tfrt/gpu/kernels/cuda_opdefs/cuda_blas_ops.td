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
// CUDA based CUDA pperation definitions.
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

def BlasCreateOp : CUDA_Op<"blas.create"> {
  let summary = "cuda blas.create operation";
  let description = [{
    cuda.blas.create creates a handle to the BLAS library context

    This will be explicitly passed to every subsequent BLAS library call.
  }];
  let arguments = (ins ContextType);
  let results = (outs CublasHandleType);
}

def BlasSaxpyOp : CUDA_Op<"blas.axpy.f32"> {
  let summary = "cuda blas.axpy.f32 operation";
  let description = [{
    cuda.blas.axpy.f32 call BLAS saxpy function.
  }];
  let arguments = (ins ContextType, CublasHandleType, I32:$n, F32:$alpha, BufferType:$x, I32:$incx, BufferType:$y, I32:$incy, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

#endif  // CUDA_OPS
