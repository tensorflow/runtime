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

// Helper functions for Tensorflow broadcasting rules.

#include "tfrt/common/ops/tf/bcast.h"

#include <sys/types.h>

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

Expected<TensorShape> GetBroadcastedShape(const TensorShape& arg0_shape,
                                          const TensorShape& arg1_shape) {
  const int rank = std::max(arg0_shape.GetRank(), arg1_shape.GetRank());

  SmallVector<ssize_t, 8> ret;
  ret.reserve(rank);

  auto dim = [](const TensorShape& shape, int i) -> ssize_t {
    if (i >= shape.GetRank()) return 1;
    return shape.GetDimensionSize(shape.GetRank() - i - 1);
  };

  for (int i = 0; i < rank; i++) {
    const ssize_t arg0_dim = dim(arg0_shape, i);
    const ssize_t arg1_dim = dim(arg1_shape, i);

    if (arg0_dim == arg1_dim) {
      ret.push_back(arg0_dim);
    } else if (arg0_dim == 1) {
      ret.push_back(arg1_dim);
    } else if (arg1_dim == 1) {
      ret.push_back(arg0_dim);
    } else {
      return MakeStringError("Dimensions must be equal, but are ", arg0_dim,
                             " and ", arg1_dim);
    }
  }

  std::reverse(ret.begin(), ret.end());
  return TensorShape(ret);
}

Expected<ArgumentBCast> GetArgumentBCast(const TensorShape& argument_shape,
                                         const TensorShape& result_shape) {
  // It's impossible to broadcast and shrink the number of dimensions.
  if (argument_shape.GetRank() > result_shape.GetRank()) {
    return MakeStringError("argment rank must be smaller than the result rank");
  }

  size_t rank = std::max(argument_shape.GetRank(), result_shape.GetRank());

  SmallVector<ssize_t, 8> arg_dims(rank, 1);
  SmallVector<ssize_t, 8> res_dims(rank, 1);

  // Get dimensions skipping additional dimensions of size 1.
  size_t arg_extra_dims = rank - argument_shape.GetRank();
  argument_shape.GetDimensions(MutableArrayRef<ssize_t>(
      arg_dims.data() + arg_extra_dims, argument_shape.GetRank()));

  // Result can't have extra dimensions.
  result_shape.GetDimensions(&res_dims);

  // Compute a broadcast specification for the argument shape.
  SmallVector<ssize_t, 8> broadcast;
  for (size_t i = 0; i < rank; ++i) {
    if (arg_dims[i] == res_dims[i]) {
      broadcast.push_back(1);
    } else if (arg_dims[i] == 1) {
      broadcast.push_back(res_dims[i]);
    } else {
      return MakeStringError("Can't broadcast argument dim: index=", i,
                             " arg=", arg_dims[i], " res=", res_dims[i]);
    }
  }

  return ArgumentBCast(arg_dims, broadcast);
}

}  // namespace tfrt
