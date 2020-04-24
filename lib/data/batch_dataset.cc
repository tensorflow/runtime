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

//===- batch_dataset.cc ---------------------------------------------------===//
//
// This file implements BatchDatasetHelper class to batch an array of tensors
// into one tensor.
//
//===----------------------------------------------------------------------===//

#include "batch_dataset.h"

namespace tfrt {
namespace data {

template <>
llvm::Expected<DenseHostTensor> BatchValues<DenseHostTensor>(
    ArrayRef<DenseHostTensor> values, HostAllocator* allocator) {
  // Verify that all batched tensors have the same dtype and tensor shape.
  for (int i = 1, e = values.size(); i < e; i++) {
    assert((values[0].metadata() == values[i].metadata()) &&
           "tensors to be batched must have the same metadata");
  }
  // Construct the output tensor with the +1 dimension and the same dtype as
  // the batched tensors.
  SmallVector<ssize_t, 4> output_dims;
  values[0].shape().GetDimensions(&output_dims);
  const int original_rank = output_dims.size();
  output_dims.resize(original_rank + 1);
  for (int i = original_rank; i > 0; i--) {
    output_dims[i] = output_dims[i - 1];
  }
  output_dims[0] = values.size();
  TensorMetadata output_metadata(values[0].metadata().dtype, output_dims);

  auto output_dht =
      DenseHostTensor::CreateUninitialized(output_metadata, allocator);
  if (!output_dht) {
    return MakeStringError("failed to create uninitialized tensor");
  }
  // IDEA(donglin): We can optimize performance by returning a tensor backed by
  // the data of the batched tensors without copying data.
  // Copy data from the batched tensors to the output tensor
  size_t data_size = values[0].DataSizeInBytes();
  for (int i = 0; i < values.size(); i++) {
    char* ptr = static_cast<char*>(output_dht->data()) + i * data_size;
    std::memcpy(ptr, values[i].data(), data_size);
  }

  return std::move(*output_dht);
}

}  // namespace data
}  // namespace tfrt
