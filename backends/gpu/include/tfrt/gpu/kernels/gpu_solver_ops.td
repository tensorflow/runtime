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

//===- gpu_solver_ops.td -------------------------------------------------===//
//
// CUDA (cu)SOLVER based operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_SOLVER_OPS
#else
#define GPU_SOLVER_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

def GPU_SolverHandleType : GPU_Type<"SolverHandle"> {
  let mnemonic = "solver.handle";
}
def GPU_AnySolverDataType : AnyTypeOf<[AnyFloat, AnyComplex, AnySignedInteger,
                                   AnyUnsignedInteger]>;

def SolverDnCreateOp : GPU_Op<"solver.create"> {
  let description = [{
    tfrt_gpu.solver.create creates a handle to the SOLVER library context

    This will be explicitly passed to every subsequent SOLVER library call.
  }];
  let arguments = (ins GPU_StreamType);
  let results = (outs GPU_SolverHandleType);
}

def SolverDnSpotrfBufferSizeOp : GPU_Op<"solver.dn.s.portf.buffer_size"> {
  let description = [{
    tfrt_gpu.solver.dn.s.portf.buffer_size These helper kernels calculate the
                                            necessary size of work buffers for
                                            solver.dn.s.portf.
                                            "s" stands for the float type.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda, TFRT_ChainType);
  let results = (outs I32:$Lwork);
}

def SolverDnDpotrfBufferSizeOp : GPU_Op<"solver.dn.d.portf.buffer_size"> {
  let description = [{
    tfrt_gpu.solver.dn.d.portf.buffer_size These helper kernels calculate the
                                            necessary size of work buffers for
                                            solver.dn.d.portf.
                                            "d" stands for the double type.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda, TFRT_ChainType);
  let results = (outs I32:$Lwork);
}

def SolverDnCpotrfBufferSizeOp : GPU_Op<"solver.dn.c.portf.buffer_size"> {
  let description = [{
    tfrt_gpu.solver.dn.c.portf.buffer_size These helper kernels calculate the
                                            necessary size of work buffers for
                                            solver.dn.c.portf.
                                            "c" stands for the Complex type.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda, TFRT_ChainType);
  let results = (outs I32:$Lwork);
}

def SolverDnZpotrfBufferSizeOp : GPU_Op<"solver.dn.z.portf.buffer_size"> {
  let description = [{
    tfrt_gpu.solver.dn.z.portf.buffer_size These helper kernels calculate the
                                            necessary size of work buffers for
                                            solver.dn.z.portf.
                                            "z" stands for the DoubleComplex
                                            type.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda, TFRT_ChainType);
  let results = (outs I32:$Lwork);
}

def SolverDnSpotrfOp : GPU_Op<"solver.dn.s.portf"> {
  let description = [{
    tfrt_gpu.solver.dn.s.portf This kernel computes the Cholesky factorization
                                of a Hermitian positive-definite matrix. Data
                                types are real valued single precision.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda,
                   GPU_BufferType:$Workspace,
                   I32:$Lwork,
                   GPU_BufferType:$devInfo,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def SolverDnDpotrfOp : GPU_Op<"solver.dn.d.portf"> {
  let description = [{
    tfrt_gpu.solver.dn.d.portf This kernel computes the Cholesky factorization
                                of a Hermitian positive-definite matrix. Data
                                types are real valued double precision.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda,
                   GPU_BufferType:$Workspace,
                   I32:$Lwork,
                   GPU_BufferType:$devInfo,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def SolverDnCpotrfOp : GPU_Op<"solver.dn.c.portf"> {
  let description = [{
    tfrt_gpu.solver.dn.c.portf This kernel computes the Cholesky factorization
                                of a Hermitian positive-definite matrix. Data
                                types are complex valued single precision.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda,
                   GPU_BufferType:$Workspace,
                   I32:$Lwork,
                   GPU_BufferType:$devInfo,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def SolverDnZpotrfOp : GPU_Op<"solver.dn.z.portf"> {
  let description = [{
    tfrt_gpu.solver.dn.z.portf This kernel computes the Cholesky factorization
                                of a Hermitian positive-definite matrix. Data
                                types are complex valued double precision.
  }];
  let arguments = (ins GPU_SolverHandleType:$handle,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda,
                   GPU_BufferType:$Workspace,
                   I32:$Lwork,
                   GPU_BufferType:$devInfo,
                   TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

#endif  // GPU_SOLVER_OPS
