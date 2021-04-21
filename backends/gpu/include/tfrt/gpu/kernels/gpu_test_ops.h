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

// MLIR op defs for testing cuda library
//
// This file declares test-only operations for CUDA

#ifndef TFRT_GPU_KERNELS_CUDA_OPDEFS_GPU_TEST_OPS_H_
#define TFRT_GPU_KERNELS_CUDA_OPDEFS_GPU_TEST_OPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "tfrt/tensor/opdefs/tensor.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"

using namespace mlir;

namespace tfrt {
namespace gpu {

// Dialect for gpu_test operations.
class GpuTestDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tfrt_gpu_test"; }
  explicit GpuTestDialect(MLIRContext* context);
};

}  // namespace gpu
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/gpu_test_opdefs.h.inc"

#endif  // TFRT_GPU_KERNELS_CUDA_OPDEFS_GPU_TEST_OPS_H_
