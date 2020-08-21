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

//===- dense_host_tensor.cc -----------------------------------------------===//
//
// This file implements DenseHostTensor.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/dense_host_tensor.h"

#include <cstddef>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {

llvm::Optional<DenseHostTensor> DenseHostTensor::CreateUninitialized(
    const TensorMetadata& metadata, HostAllocator* allocator) {
  auto& shape = metadata.shape;
  auto data = HostBuffer::CreateUninitialized(
      metadata.dtype.GetHostSize() * shape.GetNumElements(),
      metadata.dtype.GetHostAlignment(), allocator);
  if (!data) return llvm::None;
  return DenseHostTensor(metadata, std::move(data));
}

llvm::Optional<DenseHostTensor> DenseHostTensor::CreateUninitialized(
    const TensorMetadata& metadata, HostContext* host) {
  return CreateUninitialized(metadata, host->allocator());
}

AsyncValueRef<DenseHostTensor> DenseHostTensor::MakeConstructedAsyncValueRef(
    const TensorMetadata& metadata, HostContext* host) {
  auto dht = CreateUninitialized(metadata, host);
  if (!dht) return {};

  return tfrt::MakeConstructedAsyncValueRef<DenseHostTensor>(
      host, std::move(dht.getValue()));
}

AsyncValueRef<HostTensor> DenseHostTensor::ConvertToHostTensor(
    HostContext* host, uint32_t allowed_formats) const {
  // We need to make a copy of the data, because the source and result
  // buffers are logically independent.
  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);

  auto result_alloc = CreateUninitialized(metadata(), host);
  if (!result_alloc)
    return MakeErrorAsyncValueRef(host, "out of memory copying tensor");

  auto& result_tensor = result_alloc.getValue();

  // TODO(tfrt-devs): This could be done in parallel in the background for
  // large tensors.  We could also detect when the tensor is full of broadcasted
  // data and convert to ScalarHostTensor.

  // Copy over the data.
  memcpy(result_tensor.data(), data(), DataSizeInBytes());

  result.emplace(std::move(result_tensor));
  return result;
}

void DenseHostTensor::Print(raw_ostream& os) const {
  os << "DenseHostTensor dtype = " << dtype() << ", shape = " << shape();

  auto element_size = dtype().GetHostSize();
  auto* data_ptr = static_cast<const char*>(data());

  static const ssize_t kThreshold = 32;
  if (NumElements() > kThreshold) {
    // Print a MD5 sum of the tensor contents.
    os << ", md5sum = ";
    uint32_t md5sum =
        llvm::MD5Hash(llvm::StringRef(data_ptr, DataSizeInBytes()));
    os << md5sum;
  }

  // Print at most 32 elements for a tensor.
  os << ", values = [";
  for (ssize_t i = 0, e = std::min(kThreshold, NumElements()); i != e; ++i) {
    if (i != 0) os << ", ";
    dtype().Print(data_ptr + i * element_size, os);
  }
  if (NumElements() > kThreshold) {
    os << ", ... ";
  }
  os << ']';
}

}  // namespace tfrt
