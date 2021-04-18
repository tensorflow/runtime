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

//===- cuda_solver_ops.td ---------------------------------------------------------===//
//
// CUDA (cu)SOLVER based operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_SOLVER_OPS
#else
#define CUDA_SOLVER_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td"

def GPU_SolverHandleType : GPU_Type<"solver_handle">;
def GPU_AnySolverDataType : AnyTypeOf<[AnyFloat, AnyComplex, AnySignedInteger,
                                   AnyUnsignedInteger]>;

def SolverDnCreateOp : GPU_Op<"solver.create"> {
  let summary = "tfrt_cuda solver.create operation";
  let description = [{
    tfrt_cuda.solver.create creates a handle to the SOLVER library context

    This will be explicitly passed to every subsequent SOLVER library call.
  }];
  let arguments = (ins GPU_ContextType, TFRT_ChainType);
  let results = (outs GPU_SolverHandleType);
}

def SolverDnSetStreamOp : GPU_Op<"solver.set_stream"> {
  let summary = "tfrt_cuda solver.set_stream operation";
  let description = [{
    tfrt_cuda.solver.set_stream sets the SOLVER library stream.

    All subsequent SOLVER calls will execute on the set stream (until
    solver.set_stream is called with a different stream).
  }];
  let arguments = (ins GPU_SolverHandleType, GPU_StreamType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def SolverDnSpotrfBufferSizeOp : GPU_Op<"solver.dn.s.portf.buffer_size"> {
  let summary = "tfrt_cuda solver.dn.s.portf.buffer_size operation";
  let description = [{
    tfrt_cuda.solver.dn.s.portf.buffer_size These helper kernels calculate the
                                            necessary size of work buffers for
                                            solver.dn.s.portf.
  }];
  let arguments = (ins GPU_ContextType,
                   GPU_SolverHandleType,
                   I32:$uplo,
                   I32:$n,
                   GPU_BufferType:$A,
                   I32:$lda, TFRT_ChainType);
  let results = (outs I32:$Lwork);
}

def SolverDnSpotrfOp : GPU_Op<"solver.dn.s.portf"> {
  let summary = "tfrt_cuda solver.dn.s.portf operation";
  let description = [{
    tfrt_cuda.solver.dn.s.portf This kernel computes the Cholesky factorization
                                of a Hermitian positive-definite matrix. Data
                                types are real valued single precision.
  }];
  let arguments = (ins GPU_ContextType,
                   GPU_SolverHandleType,
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

#endif  // CUDA_SOLVER_OPS
