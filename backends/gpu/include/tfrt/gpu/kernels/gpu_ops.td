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

include "mlir/Dialect/GPU/IR/GPUBase.td"

def TFRTGPU_StreamifyOp : TFRTGPU_Op<"streamify", [
  GPU_AsyncOpInterface,
  SingleBlockImplicitTerminator<"::tfrt::compiler::ReturnOp">
]> {
  let summary = "tfrt_gpu.streamify operation";
  let description = [{
    tfrt_gpu.streamify wraps a block to take and produce !gpu.async.tokens
    (i.e. it implements the mlir::gpu::AsyncOpInterface).

    This op is used only during conversion to the tfrt_gpu dialect and has no
    kernel implementation.

    The region has a single block with !tfrt.chain and !tfrt_gpu.stream
    arguments and at least a !tfrt.chain return operand. Ops within the region
    can be lowered to tfrt_gpu ops before the surrounding code. Usually such
    lowering is implemented by deriving from GpuStreamifyOpConversionPattern<T>.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    result_range results();
    OpResult asyncToken();
    OpResult getAsyncToken() { return asyncToken(); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"::mlir::ValueRange", "::std::nullopt">:$results)>
  ];

  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif  // TFRTGPU_OPS
