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

// Helpers for DNN ops.

#ifndef TFRT_BACKENDS_COMMON_OPS_TF_DNN_OPS_UTIL_H_
#define TFRT_BACKENDS_COMMON_OPS_TF_DNN_OPS_UTIL_H_

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace llvm {
class raw_ostream;
}  // namespace llvm

namespace tfrt {
class OpAttrsRef;
class TensorShape;

llvm::SmallVector<Index, 4> GetDimensions(const TensorShape& shape);

void RotateRight(llvm::MutableArrayRef<Index> array, size_t k = 1);

enum class ChannelOrder { ChannelFirst, ChannelLast };

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              ChannelOrder channel_order);

// Extracts channel order from 'data_format' attribute. Defaults to ChannelLast.
ChannelOrder GetTfChannelOrder(std::optional<string_view> data_format);

struct WindowedOutputData {
  llvm::SmallVector<Index, 4> output_dims;  // NCHW
  llvm::SmallVector<Index, 2> strides;
  llvm::SmallVector<Index, 2> dilations;
  llvm::SmallVector<Index, 2> paddings_before;
  llvm::SmallVector<Index, 2> paddings_after;
};

// Get the output shape, strides, dilations and padding of conv-like ops.
llvm::Expected<WindowedOutputData> GetTfWindowedOutputData(
    llvm::ArrayRef<Index> input_dims,   // NCHW
    llvm::ArrayRef<Index> filter_dims,  // OIHW
    ChannelOrder channel_order, string_view padding_string,
    ArrayRef<int> explicit_paddings, ArrayRef<Index> strides,
    ArrayRef<Index> dilations);

// Expands 'sizes' to 'rank' length, inserting ones.
//
// The length of 'sizes' can be 1, rank-2, or rank. For the latter, the elements
// are in 'channel_order'. The result is always NCHW order.
llvm::SmallVector<Index, 4> MaybeExpandFilterSizes(llvm::ArrayRef<Index> sizes,
                                                   int rank,
                                                   ChannelOrder channel_order);

// Compute the output shape for a Pad op.
template <typename Tpadding>
Expected<TensorMetadata> TfPadOutputShape(
    const TensorMetadata& input, const DenseTensorView<Tpadding, 2>& paddings) {
  const TensorShape& input_shape = input.shape;
  llvm::SmallVector<Index, 8> output_shape;
  for (int d = 0; d < input_shape.GetRank(); ++d) {
    const Tpadding before_d =
        paddings.GetElementAt(d, 0);  // Pad before existing elements.
    const Tpadding after_d =
        paddings.GetElementAt(d, 1);  // Pad after existing elements.
    if (before_d < 0 || after_d < 0) {
      return MakeStringError("tf.Pad paddings must be non-negative.");
    }
    const Index size_d = input_shape.GetDimensionSize(d);
    output_shape.push_back(before_d + size_d + after_d);
  }
  return TensorMetadata(input.dtype, output_shape);
}

// Guess channel order from two dimensions being the same.
// TODO(tfrt-devs): Fix call sites and remove.
std::optional<ChannelOrder> GuessChannelOrder(const TensorShape& shape);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_OPS_TF_DNN_OPS_UTIL_H_
