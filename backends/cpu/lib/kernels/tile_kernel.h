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

//===- tile_kernel.h --------------------------------------------*- C++ -*-===//
//
// Tensorflow Tile kernel implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_TILE_KERNEL_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_TILE_KERNEL_H_

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace cpu {

static Expected<SmallVector<ssize_t, 5>> TileMultiples(
    const DenseHostTensor& multiples_arg) {
  SmallVector<ssize_t, 5> multiples;

  if (multiples_arg.shape().GetRank() != 1) {
    return MakeStringError("Tile multiples must be a vector");
  }

  if (multiples_arg.dtype().kind() == DType::I32) {
    DHTArrayView<int32_t> view(&multiples_arg);
    auto els = view.Elements();
    for (int i = 0; i < view.NumElements(); ++i) multiples.push_back(els[i]);

  } else if (multiples_arg.dtype().kind() == DType::I64) {
    DHTArrayView<int64_t> view(&multiples_arg);
    auto els = view.Elements();
    for (int i = 0; i < view.NumElements(); ++i) multiples.push_back(els[i]);

  } else {
    return MakeStringError("Unsupported multiples data type");
  }

  return multiples;
}

template <int rank>
struct TileRankTag {
  static constexpr int value = rank;
};

template <typename T>
static AsyncValueRef<Chain> Tile(const DenseHostTensor& input,
                                 const SmallVector<ssize_t, 5>& multiples,
                                 DenseHostTensor* output,
                                 const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto& ctx = host->GetOrCreateSharedContext<compat::EigenHostContext>();

  auto rank_dispatch = [&](auto rank_tag) -> AsyncValueRef<Chain> {
    static constexpr int rank = decltype(rank_tag)::value;

    auto input_view = DHTIndexableView<T, rank>(&input);
    auto output_view = MutableDHTIndexableView<T, rank>(output);

    auto input_t = compat::AsEigenConstTensor(input_view);
    auto output_t = compat::AsEigenTensor(output_view);

    Eigen::DSizes<Eigen::Index, rank> broadcast;
    for (int i = 0; i < rank; ++i) {
      broadcast[i] = multiples[i];
    }

    return AsyncAssign(ctx, output_t, input_t.broadcast(broadcast),
                       compat::KeepBuffers::alive(&input, output));
  };

  // Dispatch based on the output tensor rank.
  const int rank = output->shape().GetRank();

  if (rank == 1) {
    return rank_dispatch(TileRankTag<1>{});
  } else if (rank == 2) {
    return rank_dispatch(TileRankTag<2>{});
  } else if (rank == 3) {
    return rank_dispatch(TileRankTag<3>{});
  } else if (rank == 4) {
    return rank_dispatch(TileRankTag<4>{});
  } else if (rank == 5) {
    return rank_dispatch(TileRankTag<5>{});
  } else {
    return host->MakeErrorAsyncValueRef("Unsupported tensor rank");
  }
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_TILE_KERNEL_H_
