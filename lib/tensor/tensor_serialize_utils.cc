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

//===- tensor_serialize_utils.cc ------------------------------------------===//
//
// This file defines serialization and deserialization utils for DenseElement
// attributes.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/tensor_serialize_utils.h"

#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/byte_order.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_view.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// TODO(tfrt-devs): Consider creating a custom buffer with 8-byte
// alignment for the tensor data instead of using std::vector<uint64_t>.
std::vector<uint8_t> SerializeDenseHostTensorToDenseAttr(
    const DenseHostTensor& dht) {
  std::vector<uint8_t> data;

  const auto& md = dht.metadata();
  const auto& buf = *dht.buffer();

  BEFDenseAttr header;
  header.base.type = GetDenseAttributeType(md.dtype.kind());
  header.rank = AssertAttrFieldSize16(md.shape.GetRank());
  header.num_elements = AssertAttrFieldSize32(md.shape.GetNumElements());

  header.shape_offset = AssertAttrFieldSize16(
      llvm::alignTo(sizeof(BEFDenseAttr), alignof(int64_t)));
  data.resize(header.shape_offset, 0xCC);

  SmallVector<int64_t, 4> shape;
  for (int i = 0; i < header.rank; ++i) {
    shape.push_back(md.shape.GetDimensionSize(i));
  }

  auto shape_buffer =
      llvm::makeArrayRef(reinterpret_cast<const uint8_t*>(shape.data()),
                         shape.size() * sizeof(int64_t));
  data.insert(data.end(), shape_buffer.begin(), shape_buffer.end());

  // Always align element data to 8-byte boundary.
  header.element_offset = llvm::alignTo(data.size(), 8);
  data.resize(header.element_offset);

  auto elements = llvm::makeArrayRef(
      reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
  data.insert(data.end(), elements.begin(), elements.end());
  SetBEFAttrByteCount(data.size(), &header.base);

  std::memcpy(data.data(), &header, sizeof(BEFDenseAttr));

  return data;
}

llvm::Expected<DenseHostTensor> DeserializeDenseHostTensorFromDenseAttr(
    DenseAttr attr, HostContext* host) {
  TensorMetadata md(DType(attr.dtype()), attr.shape());

  auto result_alloc = DenseHostTensor::CreateUninitialized(md, host);
  if (!result_alloc) {
    return MakeStringError("error creating DenseHostTensor");
  }

  auto& result_tensor = result_alloc.getValue();
  std::memcpy(result_tensor.data(), attr.GetElements(),
              result_tensor.DataSizeInBytes());
  return std::move(result_tensor);
}

TensorMetadata CreateTensorMetadata(const DenseAttr& attr) {
  return CreateDenseView(attr).metadata();
}

DenseView CreateDenseView(const DenseAttr& attr) {
  return DenseView(DType(attr.dtype()), attr.shape(), attr.GetElements());
}

// Write value to location in little endian manner.
char* WriteUint64(uint64_t value, char* location) {
  uint8_t data[] = {
      uint8_t(value & 0xFF),         uint8_t((value >> 8) & 0xFF),
      uint8_t((value >> 16) & 0xFF), uint8_t((value >> 24) & 0xFF),
      uint8_t((value >> 32) & 0xFF), uint8_t((value >> 40) & 0xFF),
      uint8_t((value >> 48) & 0xFF), uint8_t((value >> 56) & 0xFF)};
  memcpy(location, data, sizeof(uint64_t));
  return location + sizeof(uint64_t);
}

std::string SerializeTensorMetadata(const TensorMetadata& md) {
  std::string buffer;
  buffer.resize(sizeof(uint64_t) * (md.shape.GetRank() + 1));
  char* pos = &buffer[0];
  pos = WriteUint64(static_cast<uint64_t>(md.dtype.kind()), pos);
  SmallVector<ssize_t, 4> dimensions;
  md.shape.GetDimensions(&dimensions);
  for (int i = 0; i < dimensions.size(); ++i) {
    pos = WriteUint64(dimensions[i], pos);
  }
  return buffer;
}

llvm::Expected<TensorMetadata> DeserializeTensorMetadata(
    string_view serialized) {
  ASSERT_LITTLE_ENDIAN();
  const char* pos = serialized.data();
  DType::Kind kind =
      static_cast<DType::Kind>(*reinterpret_cast<const uint64_t*>(pos));
  pos += sizeof(uint64_t);

  const int num_elements = serialized.size() / 8 - 1;
  SmallVector<ssize_t, 4> dimensions;
  dimensions.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    dimensions.push_back(*reinterpret_cast<const uint64_t*>(pos));
    pos += sizeof(uint64_t);
  }

  TensorShape shape(dimensions);
  TensorMetadata md(DType(kind), shape);

  return md;
}

}  // namespace tfrt
