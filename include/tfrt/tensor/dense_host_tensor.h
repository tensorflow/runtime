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

// This file defines the DenseHostTensor class.

#ifndef TFRT_TENSOR_DENSE_HOST_TENSOR_H_
#define TFRT_TENSOR_DENSE_HOST_TENSOR_H_

#include <optional>

#include "tfrt/host_context/host_buffer.h"
#include "tfrt/host_context/value.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
class HostContext;

void RegisterDenseHostTensorConversionFn(TensorConversionFnRegistry* registry);

// Represents a tensor whose elements are stored contiguously in row major
// format with no padding or stride.
class DenseHostTensor final : public HostTensor,
                              public TensorTraits<DenseHostTensor> {
 public:
  DenseHostTensor() = default;

  DenseHostTensor(const TensorMetadata& metadata, RCReference<HostBuffer> data)
      : HostTensor(metadata), data_(std::move(data)) {}

  // Move operations are supported.
  DenseHostTensor(DenseHostTensor&& other) = default;
  DenseHostTensor& operator=(DenseHostTensor&& other) = default;

  // Allocate a DenseHostTensor with an uninitialized body.  This returns None
  // on allocation failure.
  static std::optional<DenseHostTensor> CreateUninitialized(
      const TensorMetadata& metadata, HostContext* host);

  static std::optional<DenseHostTensor> CreateUninitialized(
      const TensorMetadata& metadata, HostAllocator* allocator);

  template <typename T>
  static std::optional<DenseHostTensor> CreateUninitialized(
      const TensorShape& shape, HostContext* host) {
    return CreateUninitialized(TensorMetadata(GetDType<T>(), shape), host);
  }

  template <typename T>
  static std::optional<DenseHostTensor> CreateScalar(T value,
                                                     HostContext* host) {
    auto dht_or = CreateUninitialized(TensorMetadata(GetDType<T>(), {}), host);
    if (!dht_or.has_value()) return dht_or;
    *dht_or.value().data<T>() = value;
    return dht_or;
  }

  // Make an AsyncValueRef<DenseHostTensor> with kConstructed state. This
  // returns an empty (default constructed) AsyncValueRef<T> on allocation
  // failure.
  static AsyncValueRef<DenseHostTensor> MakeConstructedAsyncValueRef(
      const TensorMetadata& metadata, HostContext* host);

  const void* data() const {
    assert(data_ && "dereferencing a null host tensor");
    return data_->data();
  }

  void* data() {
    assert(data_ && "dereferencing a null host tensor");
    return data_->data();
  }

  template <typename DType>
  const DType* data() const {
    assert(GetDType<DType>() == dtype() && "Incorrect dtype for tensor");
    return reinterpret_cast<const DType*>(data());
  }

  template <typename DType>
  DType* data() {
    assert(GetDType<DType>() == dtype() && "Incorrect dtype for tensor");
    return reinterpret_cast<DType*>(data());
  }

  const RCReference<HostBuffer>& buffer() const { return data_; }

  size_t DataSizeInBytes() const {
    if (!data_)
      return 0;
    else
      return data_->size();
  }

  RCReference<HostBuffer> ReleaseBuffer() { return std::move(data_); }

  DenseHostTensor CopyRef() const { return DenseHostTensor(metadata(), data_); }

  void Print(raw_ostream& os) const override;

  // Print the MD5 sum of the tensor contents. Calculating MD5 is expensive
  // so this should be used for testing only.
  void PrintMd5(raw_ostream& os) const;

  // Tensor type for DenseHostTensor.
  static const char* name() { return "DenseHost"; }

 private:
  // This class is not copyable or assignable. If we add a copy operation it
  // will likely be explicit.
  DenseHostTensor(const DenseHostTensor& other) = delete;
  DenseHostTensor& operator=(const DenseHostTensor&) = delete;

  RCReference<HostBuffer> data_;
};

// This is to ensure that Value can store DenseHostTensor without heap
// allocation. This limits the size of DenseHostTensor to at most 56 bytes.
static_assert(Value::IsInPlace<DenseHostTensor>(),
              "DenseHostTensor should not cause a heap allocation in Value.");

// Compares the metadata and underlying byte buffers for equality. Please see
// `TensorApproxEqual` for tensors with floating point numbers.
bool operator==(const DenseHostTensor& a, const DenseHostTensor& b);

inline bool operator!=(const DenseHostTensor& a, const DenseHostTensor& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& o, const DenseHostTensor& dht);

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_HOST_TENSOR_H_
