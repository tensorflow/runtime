/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MLIR op definitions for cuda_ops library
//
// This file declares the 'cuda' dialect as well as the operators that make up
// the cuda_ops library.

#ifndef TFRT_GPU_KERNELS_CUDA_OPDEFS_CUDA_OPS_H_
#define TFRT_GPU_KERNELS_CUDA_OPDEFS_CUDA_OPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "tfrt/tensor/opdefs/host_tensor.h"
#include "tfrt/tensor/opdefs/tensor.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"

using namespace mlir;

namespace tfrt {
namespace cuda {

// Dialect for cuda operations.
class CUDADialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tfrt_cuda"; }
  explicit CUDADialect(MLIRContext* context);
};

namespace conversion {

// Dialect for cuda conversion helper operations.
class CUDA_ConversionDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tfrt_cuda_conversion"; }
  explicit CUDA_ConversionDialect(MLIRContext* context);
};

}  // namespace conversion

}  // namespace cuda
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_opdefs.h.inc"

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_conversion_helper_opdefs.h.inc"

#endif  // TFRT_GPU_KERNELS_CUDA_OPDEFS_CUDA_OPS_H_
