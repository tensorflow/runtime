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

// This file defines serialization and deserialization utils for DenseElement
// attributes.

#include "tfrt/tensor/tensor_serialize_utils.h"

#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/host_context.h"
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

void SerializeTensorMetadataInternal(const TensorMetadata& md, char* pos) {
  pos = WriteUint64(static_cast<uint64_t>(md.dtype.kind()), pos);
  SmallVector<ssize_t, 4> dimensions;
  md.shape.GetDimensions(&dimensions);
  for (int i = 0; i < dimensions.size(); ++i) {
    pos = WriteUint64(dimensions[i], pos);
  }
}

std::string SerializeTensorMetadata(const TensorMetadata& md) {
  std::string buffer;
  buffer.resize(sizeof(uint64_t) * (md.shape.GetRank() + 1));
  SerializeTensorMetadataInternal(md, &buffer[0]);
  return buffer;
}

llvm::Expected<TensorMetadata> DeserializeTensorMetadataInternal(
    const char* pos, size_t size) {
  ASSERT_LITTLE_ENDIAN();
  DType::Kind kind =
      static_cast<DType::Kind>(*reinterpret_cast<const uint64_t*>(pos));
  pos += sizeof(uint64_t);
  const int num_dimensions = size / 8 - 1;
  SmallVector<ssize_t, 4> dimensions;
  dimensions.reserve(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    dimensions.push_back(*reinterpret_cast<const uint64_t*>(pos));
    pos += sizeof(uint64_t);
  }
  TensorShape shape(dimensions);
  TensorMetadata md(DType(kind), shape);
  return md;
}

llvm::Expected<TensorMetadata> DeserializeTensorMetadata(
    string_view serialized) {
  ASSERT_LITTLE_ENDIAN();
  return DeserializeTensorMetadataInternal(serialized.data(),
                                           serialized.size());
}

llvm::Expected<llvm::SmallVector<RCReference<HostBuffer>, 4>>
SerializeDenseHostTensor(const DenseHostTensor& dht, HostContext* host) {
  SmallVector<RCReference<HostBuffer>, 4> buffers;
  // A DenseHostTensor has 2 elements: tensor metadata and tensor buffer.
  buffers.reserve(2);
  const auto& md = dht.metadata();
  // Serialized metadata consists of
  // - dimension (obtained by GetRank())
  // - tensor kind (hence +1)
  size_t md_size = sizeof(uint64_t) * (md.shape.GetRank() + 1);
  auto md_buffer = tfrt::HostBuffer::CreateUninitialized(
      /*size=*/md_size, /*alignment=*/1, host->allocator());
  if (!md_buffer) {
    return MakeStringError("error serializing DenseHostTensor");
  }
  SerializeTensorMetadataInternal(md, static_cast<char*>(md_buffer->data()));
  buffers.push_back(md_buffer.CopyRef());
  buffers.push_back(dht.buffer().CopyRef());
  return std::move(buffers);
}

llvm::Expected<DenseHostTensor> DeserializeDenseHostTensor(
    const llvm::SmallVector<RCReference<HostBuffer>, 4>& serialized,
    HostContext* host) {
  TensorMetadata md =
      DeserializeTensorMetadataInternal(
          static_cast<char*>(serialized[0]->data()), serialized[0]->size())
          .get();
  auto dht = DenseHostTensor(md, serialized[1].CopyRef());
  return std::move(dht);
}
}  // namespace tfrt
