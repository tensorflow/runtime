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

//===- string_host_tensor.h -------------------------------------*- C++ -*-===//
//
// This file defines the StringHostTensor class.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_STRING_HOST_TENSOR_H_
#define TFRT_TENSOR_STRING_HOST_TENSOR_H_

#include <string>

#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/tensor/host_tensor.h"

namespace tfrt {

// Represents a tensor of strings. The metadata of strings (pointer and size)
// are stored contiguously in row major format with no padding or stride.
class StringHostTensor final : public HostTensor,
                               public TensorTraits<StringHostTensor> {
 public:
  // Allocate a StringHostTensor with uninitialized data. Return None on
  // failure.
  static llvm::Optional<StringHostTensor> CreateUninitialized(
      const TensorMetadata& metadata, HostContext* host);

  static llvm::Optional<StringHostTensor> CreateUninitialized(
      const TensorShape& shape, HostContext* host);

  // Make an AsyncValueRef<StringHostTensor> with kConstructed state.
  static AsyncValueRef<StringHostTensor> MakeConstructedAsyncValueRef(
      const TensorMetadata& metadata, HostContext* host);

  StringHostTensor(const TensorMetadata& metadata,
                   HostArray<std::string> strings)
      : HostTensor(Subclass::StringHost, metadata),
        strings_(std::move(strings)) {
    assert(metadata.dtype == DType(DType::String));
  }

  StringHostTensor(const TensorShape& shape, HostArray<std::string> strings)
      : StringHostTensor{TensorMetadata{DType(DType::String), shape},
                         std::move(strings)} {}

  StringHostTensor(StringHostTensor&& other);
  StringHostTensor& operator=(StringHostTensor&& other);

  StringHostTensor(const StringHostTensor& other) = delete;
  StringHostTensor& operator=(const StringHostTensor& other) = delete;

  ArrayRef<std::string> strings() const { return strings_.array(); }
  MutableArrayRef<std::string> strings() { return strings_.mutable_array(); }

  void Print(raw_ostream& os) const override;

  AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, uint32_t allowed_formats) const override;

  // Tensor type for StringHostTensor.
  static const char* name() { return "StringHost"; }

 private:
  // TODO(tfrt-devs): Consider making it reference counted.
  HostArray<std::string> strings_;
};

inline StringHostTensor::StringHostTensor(StringHostTensor&& other) = default;
inline StringHostTensor& StringHostTensor::operator=(StringHostTensor&& other) =
    default;

}  // namespace tfrt

#endif  // TFRT_TENSOR_STRING_HOST_TENSOR_H_
