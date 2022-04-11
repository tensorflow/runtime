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

//===- gpu_ops.td --------------------------------------------------------===//
//
// CUDA based CUDA operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef TFRTGPU_OPS
#else
#define TFRTGPU_OPS

include "tfrt/gpu/kernels/gpu_blas_ops.td"
include "tfrt/gpu/kernels/gpu_ccl_ops.td"
include "tfrt/gpu/kernels/gpu_dnn_ops.td"
include "tfrt/gpu/kernels/gpu_driver_ops.td"
include "tfrt/gpu/kernels/gpu_fft_ops.td"
include "tfrt/gpu/kernels/gpu_solver_ops.td"

include "mlir/Dialect/GPU/GPUBase.td"

def TFRTGPU_StreamifyOp : TFRTGPU_Op<"streamify",
    [GPU_AsyncOpInterface, SingleBlockImplicitTerminator<"compiler::ReturnOp">]> {
  let summary = "tfrt_gpu.streamify operation";
  let description = [{
    tfrt_gpu.streamify wraps a block to take and produce
    !gpu.async.tokens (i.e. it implements the mlir::gpu::AsyncOpInterface).

    The region has a single block with !tfrt.chain and !gpu.async.token
    arguments and a !tfrt.chain terminator operand.

    This op is used by the Dialect Conversion framework when lowering to the
    tfrt_gpu dialect. This op should only exist during the conversion process,
    and so doesn't map to an implementation.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = ?;
}

#endif  // TFRTGPU_OPS
