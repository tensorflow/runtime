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

#include "tfrt/tensor/btf_util.h"

#include <iostream>

namespace tfrt {
namespace {

constexpr size_t kBtfAlignment = 8;

size_t Pad(size_t n) {
  const size_t remainder = n % kBtfAlignment;
  if (remainder == 0) return 0;
  return kBtfAlignment - remainder;
}

Error WriteDHTToBTF(std::ostream* stream, const DenseHostTensor& dht) {
  std::vector<uint64_t> dims;
  dims.reserve(dht.shape().GetRank());
  for (int i = 0; i < dht.shape().GetRank(); i++) {
    dims.push_back(dht.shape().GetDimensionSize(i));
  }
  if (!WriteStream(stream, dims.data(), dims.size())) {
    return MakeStringError("failed to write tensor dims");
  }
  const size_t nbytes = dht.DataSizeInBytes();
  if (!WriteStream(stream, reinterpret_cast<const uint8_t*>(dht.data()),
                   nbytes)) {
    return MakeStringError("failed to write tensor data");
  }
  const size_t padding = Pad(nbytes);
  constexpr std::array<uint8_t, 7> pad_value{0, 0, 0, 0, 0, 0, 0};
  if (padding != 0 && !WriteStream(stream, pad_value.data(), padding)) {
    return MakeStringError("failed to pad tensor");
  }
  return Error::success();
}

}  // namespace

Expected<std::vector<uint64_t>> ReadBTFOffsets(std::istream* stream) {
  uint64_t num_tensors;
  if (!ReadStream(stream, &num_tensors, 1)) {
    return MakeStringError("failed to read num_tensors");
  }
  std::vector<uint64_t> offsets;
  offsets.resize(num_tensors);
  if (!ReadStream(stream, offsets.data(), num_tensors)) {
    return MakeStringError("failed to read tensor record offsets");
  }
  return offsets;
}

Expected<DenseHostTensor> ReadDHTFromBTF(std::istream* stream, uint64_t offset,
                                         HostContext* host) {
  stream->seekg(offset);
  btf::TensorHeader header;
  if (!ReadStream(stream, &header, 1)) {
    return MakeStringError("failed to read tensor header at offset", offset);
  }
  if (header.layout != btf::TensorLayout::kRMD) {
    return MakeStringError("unexpected tensor layout ", header.layout);
  }
  SmallVector<Index, 4> dims;
  dims.resize(header.rank);
  if (!ReadStream(stream, dims.data(), header.rank)) {
    return MakeStringError("failed to read tensor dims at offset", offset);
  }
  const TensorMetadata metadata(DType(ToDTypeKind(header.dtype)),
                                TensorShape(dims));
  auto dht_or = DenseHostTensor::CreateUninitialized(metadata, host);
  if (!dht_or.hasValue()) {
    return MakeStringError("cannot allocate result tensor");
  }
  auto dht = std::move(*dht_or);
  // This can read a large amount of data from the stream. Depending on the
  // underlying file system implementation, we may need to have a more optimal
  // strategy for reading the file.
  if (!ReadStream(stream, reinterpret_cast<uint8_t*>(dht.data()),
                  dht.DataSizeInBytes())) {
    return MakeStringError("failed to read tensor data at offset", offset);
  }
  return std::move(dht);
}

Error WriteTensorsToBTF(std::ostream* stream, ArrayRef<const Tensor*> tensors) {
  const uint64_t num_tensors = tensors.size();
  if (!WriteStream(stream, &num_tensors, 1)) {
    return MakeStringError("failed to write num_tensors");
  }
  std::vector<uint64_t> offsets;
  offsets.reserve(num_tensors);
  uint64_t offset = (1 + num_tensors) * sizeof(uint64_t);
  for (const Tensor* tensor : tensors) {
    offsets.push_back(offset);
    if (tensor->tensor_type() == DenseHostTensor::kTensorType) {
      const auto& dht = reinterpret_cast<const DenseHostTensor&>(*tensor);
      const size_t nbytes = dht.DataSizeInBytes();
      offset += sizeof(btf::TensorHeader) +
                dht.shape().GetRank() * sizeof(uint64_t) + nbytes + Pad(nbytes);
    } else {
      return MakeStringError("unhandled tensor type when writing btf");
    }
  }
  if (!WriteStream(stream, offsets.data(), offsets.size())) {
    return MakeStringError("failed to write offsets");
  }
  for (const Tensor* tensor : tensors) {
    auto dtype_or = btf::ToTensorDType(tensor->dtype());
    if (!dtype_or) return dtype_or.takeError();
    btf::TensorHeader header;
    header.rank = static_cast<uint64_t>(tensor->shape().GetRank());
    header.dtype = *dtype_or;
    header.layout = btf::TensorLayout::kRMD;
    if (!WriteStream(stream, &header, 1)) {
      return MakeStringError("failed to write tensor header");
    }
    if (tensor->tensor_type() == DenseHostTensor::kTensorType) {
      Error e = WriteDHTToBTF(
          stream, reinterpret_cast<const DenseHostTensor&>(*tensor));
      if (e) return e;
    } else {
      return MakeStringError("unhandled tensor type when writing btf");
    }
  }
  return Error::success();
}

}  // namespace tfrt
