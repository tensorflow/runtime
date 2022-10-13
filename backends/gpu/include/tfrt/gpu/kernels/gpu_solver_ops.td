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

#ifdef TFRTGPU_SOLVER_OPS
#else
#define TFRTGPU_SOLVER_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

def TFRTGPU_SolverHandleType : TFRTGPU_Type<"SolverHandle"> {
  let mnemonic = "solver.handle";
}

def SolverCreateOp : TFRTGPU_Op<"solver.create"> {
  let description = [{
    tfrt_gpu.solver.create creates a handle to the SOLVER library context

    This will be explicitly passed to every subsequent SOLVER library call.
  }];
  let arguments = (ins TFRTGPU_ContextType);
  let results = (outs TFRTGPU_SolverHandleType);
}

def SolverPotrfBufferSizeOp : TFRTGPU_Op<"solver.potrf.buffer_size", [Pure]> {
  let description = [{
    Calculates the work buffer size for tfrt_gpu.solver.portf.
  }];
  let arguments = (ins TFRTGPU_SolverHandleType:$handle, TFRTGPU_StreamType:$stream,
                   TFRTGPU_BlasFillModeAttr:$fillMode, I32:$n,
                   TFRTGPU_BlasDataTypeAttr:$dataType, I32:$stride,
                   TFRT_ChainType:$chain);
  let results = (outs I64:$workspaceSize);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($fillMode)`,` $n`,` custom<Enum>($dataType)`,`
    $stride`,` $chain attr-dict
  }];
}

def SolverPotrfOp : TFRTGPU_Op<"solver.potrf"> {
  let description = [{
    Computes the Cholesky factorization of a Hermitian positive-definite matrix.
  }];
  let arguments = (ins TFRTGPU_SolverHandleType:$handle, TFRTGPU_StreamType:$stream,
                   TFRTGPU_BlasFillModeAttr:$fillMode, I32:$n,
                   TFRTGPU_BlasDataTypeAttr:$dataType, TFRTGPU_BufferType:$buffer,
                   I32:$stride, TFRTGPU_BufferType:$workspace, TFRTGPU_BufferType:$devInfo,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($fillMode)`,` $n`,` custom<Enum>($dataType)`,`
    $buffer`,` $stride`,` $workspace`,` $devInfo`,` $chain attr-dict
  }];
}

def SolverPotrfBatchOp : TFRTGPU_Op<"solver.potrf.batch"> {
  let description = [{
    Computes the Cholesky factorization of a Hermitian positive-definite matrix
    using the specified batch size.
  }];
  let arguments = (ins TFRTGPU_SolverHandleType:$handle, TFRTGPU_StreamType:$stream,
                   TFRTGPU_BlasFillModeAttr:$fillMode, I32:$n,
                   TFRTGPU_BlasDataTypeAttr:$dataType, TFRTGPU_BufferType:$buffer,
                   I32:$stride, TFRTGPU_BufferType:$devInfo, I32:$batch_size,
                   TFRT_ChainType:$chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $handle`,` $stream`,` custom<Enum>($fillMode)`,` $n`,` custom<Enum>($dataType)`,`
    $buffer`,` $stride`,` $devInfo`,` $batch_size`,` $chain attr-dict
  }];
}

#endif  // TFRTGPU_SOLVER_OPS
