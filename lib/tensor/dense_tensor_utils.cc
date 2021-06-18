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

#include "tfrt/tensor/dense_tensor_utils.h"

#include <array>
#include <cstring>

#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

// Returns the offset of the given coordinate in the underlying storage. If the
// coordinates are of smaller rank than the shape, the coordinates are used as a
// prefix and the missing trailing dimensions are filled with zeros.
size_t OffsetOf(const TensorShape& shape, ArrayRef<ssize_t> dims) {
  size_t offset = 0;
  size_t stride = 1;
  for (int i = shape.GetRank() - 1; i >= 0; --i) {
    if (i < dims.size()) {
      assert(dims[i] < shape.GetDimensionSize(i));
      offset += stride * dims[i];
    }
    stride *= shape.GetDimensionSize(i);
  }
  return offset;
}

}  // namespace

DenseHostTensor Chip(const DenseHostTensor& tensor, ArrayRef<ssize_t> dims) {
  assert(!dims.empty());
  const TensorMetadata& meta = tensor.metadata();
  assert(dims.size() <= meta.shape.GetRank());
  const size_t offset = OffsetOf(meta.shape, dims);
  SmallVector<ssize_t, 4> new_shape;
  new_shape.reserve(meta.shape.GetRank() - 1);
  for (int i = dims.size(); i < meta.shape.GetRank(); i++) {
    new_shape.push_back(meta.shape.GetDimensionSize(i));
  }
  const TensorMetadata new_meta(meta.dtype, new_shape);
  auto data = HostBuffer::CreateFromExternal(tensor.buffer().CopyRef(),
                                             offset * GetHostSize(meta.dtype),
                                             new_meta.GetHostSizeInBytes());
  return DenseHostTensor(new_meta, std::move(data));
}

DenseHostTensor Flatten(const DenseHostTensor& tensor) {
  std::array<ssize_t, 1> dims{tensor.metadata().shape.GetNumElements()};
  return DenseHostTensor(
      TensorMetadata(tensor.metadata().dtype, TensorShape(dims)),
      tensor.buffer().CopyRef());
}

bool CopyTo(const DenseHostTensor& src, DenseHostTensor* dst) {
  if (src.metadata() != dst->metadata()) return false;
  std::memcpy(dst->data(), src.data(), src.DataSizeInBytes());
  return true;
}

}  // namespace tfrt
