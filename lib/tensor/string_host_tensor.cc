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

//===- string_host_tensor.cc ----------------------------------------------===//
//
// This file implements StringHostTensor.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/string_host_tensor.h"

#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

llvm::Optional<StringHostTensor> StringHostTensor::CreateUninitialized(
    const TensorMetadata& metadata, HostContext* host) {
  auto num_elements = metadata.shape.GetNumElements();
  HostArray<std::string> strings(num_elements, host->allocator());
  for (auto& str : strings.mutable_array()) {
    new (&str) std::string();
  }

  return StringHostTensor(metadata, std::move(strings));
}

AsyncValueRef<StringHostTensor> StringHostTensor::MakeConstructedAsyncValueRef(
    const TensorMetadata& metadata, HostContext* host) {
  if (auto result = CreateUninitialized(metadata, host))
    return host->MakeConstructedAsyncValueRef<StringHostTensor>(
        std::move(result).getValue());

  return {};
}

AsyncValueRef<HostTensor> StringHostTensor::ConvertToHostTensor(
    HostContext* host, uint32_t allowed_formats) const {
  assert(allowed_formats &
         (1 << static_cast<uint32_t>(Tensor::Subclass::StringHost)));

  // We need to make a copy of the data, because the source and result
  // buffers are logically independent.
  auto result = host->MakeUnconstructedAsyncValueRef<StringHostTensor>();

  auto result_alloc = CreateUninitialized(metadata(), host);
  if (!result_alloc)
    return host->MakeErrorAsyncValueRef("out of memory copying tensor");

  auto& result_tensor = result_alloc.getValue();

  // Copy over the data.
  for (int i = 0; i < NumElements(); ++i) {
    result_tensor.strings()[i] = strings()[i];
  }

  result.emplace(std::move(result_tensor));
  return result;
}

void StringHostTensor::Print(raw_ostream& os) const {
  const auto& shape = this->shape();
  os << "StringHostTensor shape = " << shape;

  auto strings = this->strings();

  static constexpr size_t kThreshold = 32;
  if (NumElements() > kThreshold) {
    llvm::MD5 hash;
    for (auto& str : strings) {
      hash.update(str);
    }
    llvm::MD5::MD5Result result;
    hash.final(result);
    os << ", md5sum = " << result.low();
  }

  os << ", values = [";
  // Print at most 32 elements for a tensor.
  for (size_t i = 0, e = std::min(kThreshold, strings.size()); i != e; ++i) {
    if (i != 0) os << ", ";
    os << '"' << strings[i] << '"';
  }

  if (NumElements() > 32) {
    os << ", ... ";
  }

  os << ']';
}

}  // namespace tfrt
