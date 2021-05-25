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

// This file implements DenseHostTensor.

#include "tfrt/tensor/dense_host_tensor.h"

#include <cstddef>
#include <cstring>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/conversion_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {

// Required alignment constraint for DHT data buffer.
// Becasuse of eigen library, the alignemnt of DHT buffer should be
// larger than or equals to EIGEN_DEFAULT_ALIGN_BYTES (16).
static constexpr size_t kTensorBufferAlignment = 16;

llvm::Optional<DenseHostTensor> DenseHostTensor::CreateUninitialized(
    const TensorMetadata& metadata, HostAllocator* allocator) {
  size_t alignment =
      std::max(metadata.dtype.GetHostAlignment(), kTensorBufferAlignment);
  auto& shape = metadata.shape;
  auto data = HostBuffer::CreateUninitialized(
      metadata.dtype.GetHostSize() * shape.GetNumElements(), alignment,
      allocator);
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

std::ostream& operator<<(std::ostream& o, const DenseHostTensor& dht) {
  llvm::raw_os_ostream os(o);
  os << dht;
  return o;
}

bool operator==(const DenseHostTensor& a, const DenseHostTensor& b) {
  return a.metadata() == b.metadata() &&
         std::memcmp(a.data(), b.data(), a.metadata().GetHostSizeInBytes()) ==
             0;
}

static AsyncValueRef<DenseHostTensor> ConvertDenseHostTensorToDenseHostTensor(
    const DenseHostTensor& tensor, const CpuDevice& src, const CpuDevice& dst,
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  // We need to make a copy of the data, because the source and result
  // buffers are logically independent.
  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);

  auto result_alloc =
      DenseHostTensor::CreateUninitialized(tensor.metadata(), host);
  if (!result_alloc)
    return MakeErrorAsyncValueRef(host, "out of memory copying tensor");

  auto& result_tensor = result_alloc.getValue();

  // TODO(tfrt-devs): This could be done in parallel in the background for
  // large tensors.  We could also detect when the tensor is full of broadcasted
  // data and convert to ScalarHostTensor.

  // Copy over the data.
  memcpy(result_tensor.data(), tensor.data(), tensor.DataSizeInBytes());

  result.emplace(std::move(result_tensor));
  return result;
}

void RegisterDenseHostTensorConversionFn(TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseHostTensorToDenseHostTensor));
}

}  // namespace tfrt
