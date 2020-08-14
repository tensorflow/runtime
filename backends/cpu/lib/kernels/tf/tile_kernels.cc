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

//===- tile_kernels.cc - ----------------------------------------*- C++ -*-===//
//
// Tile Tensorflow kernels.
//
//===----------------------------------------------------------------------===//

#include "../tile_kernel.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/host_context/sync_kernel_utils.h"

namespace tfrt {
namespace {

template <typename T>
Error TileKernel(const DenseHostTensor& input,
                 const DenseHostTensor& multiples_tensor,
                 DenseHostTensor* output, const ExecutionContext& exec_ctx) {
  auto multiples = cpu::TileMultiples(multiples_tensor);

  if (!multiples) return multiples.takeError();

  assert(multiples->size() == input.shape().GetRank());

  return ::tfrt::cpu::Tile<T, compat::SyncEigenEvaluator>(input, *multiples,
                                                          output, exec_ctx);
}
}  // namespace
namespace tf {

void RegisterTileCpuKernels(KernelRegistry* registry) {
#define DTYPE_NUMERIC(ENUM)                                      \
  {                                                              \
    using CPP_TYPE = EigenTypeForDTypeKind<DType::ENUM>;         \
    registry->AddSyncKernel(                                     \
        StrCat("tf_sync.Tile.", GetDType<CPP_TYPE>().GetName()), \
        TFRT_SYNC_KERNEL(TileKernel<CPP_TYPE>));                 \
  }
#include "tfrt/dtype/dtype.def"

  registry->AddSyncKernel("tf_sync.Tile.string",
                          TFRT_SYNC_KERNEL(cpu::TileStringTensor));
}
}  // namespace tf

}  // namespace tfrt
