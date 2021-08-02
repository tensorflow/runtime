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

//===- cuda_conversion_ops.td ---------------------------------------------------------===//
//
// Operations used by the Dialect Conversion framework when lowering to the
// tfrt_gpu dialect. These ops only exist during the conversion process, and
// so don't map to kernels.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_CONVERSION_HELPER_OPS
#define GPU_CONVERSION_HELPER_OPS

include "tfrt/tfrt_op_base.td"
include "mlir/Dialect/GPU/GPUBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GPU_ConversionDialect : Dialect {
  let name = "tfrt_gpu_conversion";

  let description = [{
    The CUDA conversion dialect.

    This dialect is used by the Dialect Conversion framework when lowering to
    the tfrt_gpu dialect.
  }];

  let cppNamespace = "tfrt::gpu::conversion";
}

def GPU_CastOp : Op<GPU_ConversionDialect, "cast", [NoSideEffect]> {
  let summary = "tfrt_gpu_conversion.cast operation";
  let description = "Transient op to cast multiple inputs to one output.";
  let arguments = (ins Variadic<AnyType>);
  let results = (outs AnyType);
  let hasFolder = 1;
}

def GPU_AsyncExecuteOp : Op<GPU_ConversionDialect, "async.execute",
    [GPU_AsyncOpInterface, SingleBlockImplicitTerminator<"compiler::ReturnOp">]> {
  let summary = "tfrt_gpu_conversion.async.execute operation";
  let description = [{
    tfrt_gpu_conversion.async.execute wraps a block to take and produce
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
}

#endif  // GPU_CONVERSION_HELPER_OPS
