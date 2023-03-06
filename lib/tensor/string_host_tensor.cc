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

// This file implements StringHostTensor.

#include "tfrt/tensor/string_host_tensor.h"

#include <optional>
#include <utility>

#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/conversion_utils.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

std::optional<StringHostTensor> StringHostTensor::CreateUninitialized(
    const TensorShape& shape, HostContext* host) {
  auto num_elements = shape.GetNumElements();
  if (num_elements == 0) {
    return StringHostTensor(shape, /*strings=*/{});
  }

  HostArray<std::string> strings(num_elements, host->allocator());
  for (auto& str : strings.mutable_array()) {
    new (&str) std::string();
  }

  return StringHostTensor(shape, std::move(strings));
}

std::optional<StringHostTensor> StringHostTensor::CreateUninitialized(
    const TensorMetadata& metadata, HostContext* host) {
  assert(metadata.dtype == DType(DType::String));
  return CreateUninitialized(metadata.shape, host);
}

AsyncValueRef<StringHostTensor> StringHostTensor::MakeConstructedAsyncValueRef(
    const TensorMetadata& metadata, HostContext* host) {
  if (auto result = CreateUninitialized(metadata, host))
    return tfrt::MakeConstructedAsyncValueRef<StringHostTensor>(
        std::move(result).value());

  return {};
}

static AsyncValueRef<StringHostTensor>
ConvertStringHostTensorToStringHostTensor(const StringHostTensor& tensor,
                                          const CpuDevice& src,
                                          const CpuDevice& dst,
                                          const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  // We need to make a copy of the data, because the source and result
  // buffers are logically independent.
  auto result = MakeUnconstructedAsyncValueRef<StringHostTensor>();

  auto result_alloc = tensor.CreateUninitialized(tensor.metadata(), host);
  if (!result_alloc)
    return MakeErrorAsyncValueRef("out of memory copying tensor");

  auto& result_tensor = result_alloc.value();

  // Copy over the data.
  for (int i = 0; i < tensor.NumElements(); ++i) {
    result_tensor.strings()[i] = tensor.strings()[i];
  }

  result.emplace(std::move(result_tensor));
  return result;
}

void StringHostTensor::Print(raw_ostream& os) const {
  const auto& shape = this->shape();
  os << "StringHostTensor shape = " << shape;

  auto strings = this->strings();

  static constexpr size_t kThreshold = 16;

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

void StringHostTensor::PrintMd5(raw_ostream& os) const {
  auto strings = this->strings();
  llvm::MD5 hash;
  for (auto& str : strings) {
    hash.update(str);
  }
  llvm::MD5::MD5Result result;
  hash.final(result);
  os << ", md5sum = " << result.low();
}

HostArray<std::string> StringHostTensor::CopyBuffer(HostContext* host) const {
  HostArray<std::string> to_buffer(strings_.size(), host->allocator());

  // TODO(tfrt-dev): Consider optimizing StringHostTensor to avoid the copy
  // here.
  for (auto iter : llvm::zip(to_buffer.mutable_array(), strings())) {
    std::string& to_str = std::get<0>(iter);
    const std::string& from_str = std::get<1>(iter);
    new (&to_str) std::string(from_str);
  }

  return to_buffer;
}

void RegisterStringHostTensorConversionFn(
    TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertStringHostTensorToStringHostTensor));
}
}  // namespace tfrt
