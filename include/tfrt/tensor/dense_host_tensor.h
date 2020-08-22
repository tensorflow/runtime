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

//===- dense_host_tensor.h --------------------------------------*- C++ -*-===//
//
// This file defines the DenseHostTensor class.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_DENSE_HOST_TENSOR_H_
#define TFRT_TENSOR_DENSE_HOST_TENSOR_H_

#include "tfrt/host_context/host_buffer.h"
#include "tfrt/host_context/value.h"
#include "tfrt/tensor/host_tensor.h"

namespace tfrt {
class HostContext;

// Represents a tensor whose elements are stored contiguously in row major
// format with no padding or stride.
class DenseHostTensor final : public HostTensor,
                              public TensorTraits<DenseHostTensor> {
 public:
  DenseHostTensor(const TensorMetadata& metadata, RCReference<HostBuffer> data)
      : HostTensor(Subclass::DenseHost, metadata), data_(std::move(data)) {}

  // Move operations are supported.
  DenseHostTensor(DenseHostTensor&& other) = default;
  DenseHostTensor& operator=(DenseHostTensor&& other) = default;

  // Allocate a DenseHostTensor with an uninitialized body.  This returns None
  // on allocation failure.
  static llvm::Optional<DenseHostTensor> CreateUninitialized(
      const TensorMetadata& metadata, HostContext* host);

  static llvm::Optional<DenseHostTensor> CreateUninitialized(
      const TensorMetadata& metadata, HostAllocator* allocator);

  template <typename T>
  static llvm::Optional<DenseHostTensor> CreateUninitialized(
      const TensorShape& shape, HostContext* host) {
    return CreateUninitialized(TensorMetadata(GetDType<T>(), shape), host);
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

  const RCReference<HostBuffer>& buffer() const { return data_; }

  size_t DataSizeInBytes() const {
    if (!data_)
      return 0;
    else
      return data_->size();
  }

  RCReference<HostBuffer> ReleaseBuffer() { return std::move(data_); }

  DenseHostTensor CopyRef() const {
    return DenseHostTensor(metadata(), data_.CopyRef());
  }

  void Print(raw_ostream& os) const override;

  AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, uint32_t allowed_formats) const override;

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

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_HOST_TENSOR_H_
