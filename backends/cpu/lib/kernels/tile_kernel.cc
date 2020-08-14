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
// Tile Tensorflow kernel implementations.
//
//===----------------------------------------------------------------------===//

#include "./tile_kernel.h"

namespace tfrt {
namespace cpu {

Expected<SmallVector<ssize_t, 5>> TileMultiples(
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

void TileStringTensor(const StringHostTensor& input, StringHostTensor* output) {
  // Compute strides from the shape.
  auto strides = [](const TensorShape& shape) -> SmallVector<ssize_t, 5> {
    SmallVector<ssize_t, 5> strides(shape.GetRank());
    ssize_t stride = 1;
    for (int i = shape.GetRank() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape.GetDimensionSize(i);
    }
    return strides;
  };

  const int ndims = output->shape().GetRank();
  const ssize_t nelem = output->NumElements();

  auto in_strides = strides(input.shape());
  auto out_strides = strides(output->shape());

  ArrayRef<std::string> inp = input.strings();
  MutableArrayRef<std::string> out = output->strings();

  for (ssize_t o_idx = 0; o_idx < nelem; ++o_idx) {
    ssize_t i_idx = 0;
    ssize_t t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      ssize_t i_dim = input.shape().GetDimensionSize(i);
      i_idx += t / out_strides[i] % i_dim * in_strides[i];
      t %= out_strides[i];
    }
    out[o_idx] = inp[i_idx];
  }
}
}  // namespace cpu
}  // namespace tfrt
