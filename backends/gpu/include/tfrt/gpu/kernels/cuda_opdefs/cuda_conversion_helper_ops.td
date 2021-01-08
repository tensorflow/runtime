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
// tfrt_cuda dialect. These ops only exist during the conversion process, and
// so don't map to kernels.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_CONVERSION_HELPER_OPS
#define CUDA_CONVERSION_HELPER_OPS

include "mlir/IR/OpBase.td"
include "tfrt/tfrt_op_base.td"

def CUDA_Conversion_Dialect : Dialect {
  let name = "tfrt_cuda_conversion";

  let description = [{
    The CUDA conversion dialect.

    This dialect is used by the Dialect Conversion framework when lowering to
    the tfrt_cuda dialect.
  }];

  let cppNamespace = "tfrt::cuda::conversion";
}

def CastAnyToAnyOp : Op<CUDA_Conversion_Dialect, "cast_any_to_any"> {
  let summary = "tfrt_cuda cast_any_to_any";
  let description = [{
    tfrt_cuda.cast_any_to_any casts any type to any other type.

    This op is used by the Dialect Conversion framework when lowering to the
    tfrt_cuda dialect. This op should only exist during the conversion process,
    and so doesn't map to an implementation.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  let hasFolder = 1;
}

#endif  // CUDA_CONVERSION_HELPER_OPS
