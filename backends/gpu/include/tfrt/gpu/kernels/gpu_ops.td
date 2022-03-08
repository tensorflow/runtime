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

#ifdef GPU_OPS
#else
#define GPU_OPS

include "tfrt/gpu/kernels/gpu_blas_ops.td"
include "tfrt/gpu/kernels/gpu_ccl_ops.td"
include "tfrt/gpu/kernels/gpu_dnn_ops.td"
include "tfrt/gpu/kernels/gpu_driver_ops.td"
include "tfrt/gpu/kernels/gpu_fft_ops.td"
include "tfrt/gpu/kernels/gpu_solver_ops.td"

#endif  // GPU_OPS
