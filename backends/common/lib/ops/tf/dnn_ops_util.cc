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

// Helpers for DNN ops.
#include "tfrt/common/ops/tf/dnn_ops_util.h"

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

llvm::SmallVector<Index, 4> GetDimensions(const TensorShape& shape) {
  llvm::SmallVector<Index, 4> dimensions(shape.GetRank());
  shape.GetDimensions(&dimensions);
  return dimensions;
}

void RotateRight(llvm::MutableArrayRef<Index> array, size_t k) {
  std::rotate(array.rbegin(), array.rbegin() + k, array.rend());
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              ChannelOrder channel_order) {
  switch (channel_order) {
    case ChannelOrder::ChannelFirst:
      os << "channel_first";
      return os;
    case ChannelOrder::ChannelLast:
      os << "channel_last";
      return os;
  }
}

ChannelOrder GetTfChannelOrder(std::optional<string_view> data_format) {
  if (!data_format.has_value() || !data_format->startswith_insensitive("nc"))
    return ChannelOrder::ChannelLast;
  return ChannelOrder::ChannelFirst;
}

llvm::Expected<WindowedOutputData> GetTfWindowedOutputData(
    llvm::ArrayRef<Index> input_dims,   // NCHW
    llvm::ArrayRef<Index> filter_dims,  // OIHW
    ChannelOrder channel_order, string_view padding_string,
    ArrayRef<int> explicit_paddings, ArrayRef<Index> strides,
    ArrayRef<Index> dilations) {
  auto rank = input_dims.size();
  if (filter_dims.size() != rank)
    return MakeStringError("Input and filter must have same rank.");

  TFRT_ASSIGN_OR_RETURN(auto padding_type,
                        compat::ParsePaddingType(padding_string));

  if (padding_type == compat::PaddingType::kExplicit) {
    if (explicit_paddings.empty())
      return MakeStringError("Missing 'explicit_paddings' attribute");
    if (explicit_paddings.size() != 2 * (rank - 2))
      return MakeStringError("Wrong 'explicit_paddings' attribute length");
  }

  auto strides_expanded =
      MaybeExpandFilterSizes(strides, filter_dims.size(), channel_order);
  auto dilations_expanded =
      MaybeExpandFilterSizes(dilations, filter_dims.size(), channel_order);

  WindowedOutputData result;
  result.output_dims.push_back(input_dims[0]);
  result.output_dims.push_back(filter_dims[0]);
  result.strides.assign(strides_expanded.begin() + 2, strides_expanded.end());
  result.dilations.assign(dilations_expanded.begin() + 2,
                          dilations_expanded.end());

  std::optional<compat::Padding> paddings;
  for (int i = 2; i < filter_dims.size(); ++i) {
    if (padding_type == compat::PaddingType::kExplicit) {
      paddings = compat::Padding{explicit_paddings[0], explicit_paddings[1]};
      explicit_paddings = explicit_paddings.drop_front(2);
    }
    TFRT_ASSIGN_OR_RETURN(
        auto output_dim, compat::ComputeWindowedOutputDimension(
                             input_dims[i], filter_dims[i], strides_expanded[i],
                             dilations_expanded[i], padding_type, paddings));
    result.output_dims.push_back(output_dim.output_size);
    result.paddings_before.push_back(output_dim.padding.padding_before);
    result.paddings_after.push_back(output_dim.padding.padding_after);
  }
  return result;
}

llvm::SmallVector<Index, 4> MaybeExpandFilterSizes(llvm::ArrayRef<Index> sizes,
                                                   int rank,
                                                   ChannelOrder channel_order) {
  llvm::SmallVector<Index, 4> result(sizes.begin(), sizes.end());

  if (result.empty()) result.push_back(1);

  if (result.size() == 1) result.resize(rank - 2, result.front());

  if (result.size() == rank - 2) {
    result.resize(rank, 1);  // Add NC.
    RotateRight(result, 2);  // HWNC to NCHW.
  } else if (channel_order == ChannelOrder::ChannelLast) {
    // NHWC to NCHW.
    RotateRight(llvm::MutableArrayRef<Index>(result).drop_front());
  }
  return result;
}

std::optional<ChannelOrder> GuessChannelOrder(const TensorShape& shape) {
  auto dims = GetDimensions(shape);
  if (dims.size() != 4) return {};
  if (dims[2] == dims[3]) return ChannelOrder::ChannelFirst;
  if (dims[1] == dims[2]) return ChannelOrder::ChannelLast;
  return {};
}
}  // namespace tfrt
