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

#include "tfrt/support/byte_order.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_view.h"
#include "tfrt/tensor/dtype.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

BEFAttributeType ConvertTensorDTypeToBEFAttributeType(DType dtype) {
  switch (dtype.kind()) {
    case DType::I32:
      return BEFAttributeType::kI32;
    case DType::I64:
      return BEFAttributeType::kI64;
    case DType::F16:
      return BEFAttributeType::kF16;
    case DType::F32:
      return BEFAttributeType::kF32;
    case DType::F64:
      return BEFAttributeType::kF64;
    default:
      llvm_unreachable("unsupported dtype.");
  }
}

DType ConvertBEFAttributeTypeToTensorDType(BEFAttributeType kind) {
  switch (kind) {
    case BEFAttributeType::kI32:
      return DType(DType::I32);
    case BEFAttributeType::kI64:
      return DType(DType::I64);
    case BEFAttributeType::kF16:
      return DType(DType::F16);
    case BEFAttributeType::kF32:
      return DType(DType::F32);
    case BEFAttributeType::kF64:
      return DType(DType::F64);
    default:
      llvm_unreachable("unsupported dtype.");
  }
}

// TODO(tf-runtime-team): Consider creating a custom buffer with 8-byte
// alignment for the tensor data instead of using std::vector<uint64_t>.
void SerializeTensorMetadata(const TensorMetadata& md,
                             std::vector<uint64_t>* res) {
  BEFDenseAttrHeader header;
  header.dtype =
      static_cast<uint8_t>(ConvertTensorDTypeToBEFAttributeType(md.dtype));
  header.rank = md.shape.GetRank();
  header.size = md.shape.GetNumElements();

  const uint64_t* raw_header = reinterpret_cast<const uint64_t*>(&header);
  res->push_back(raw_header[0]);
  res->push_back(raw_header[1]);

  SmallVector<int64_t, 4> shape;
  md.shape.GetDimensions(&shape);

  for (auto dim : shape) {
    res->push_back(static_cast<uint64_t>(dim));
  }
}

void SerializeHostBuffer(const HostBuffer& buf, std::vector<uint64_t>* res) {
  auto prev_size = res->size();
  res->resize(prev_size + (buf.size() + 7) / 8, 0);
  std::memcpy(&res->at(prev_size), buf.data(), buf.size());
}

}  // namespace

std::vector<uint64_t> SerializeDenseHostTensorToDenseAttr(
    const DenseHostTensor& dht) {
  std::vector<uint64_t> data;
  data.reserve(512);
  SerializeTensorMetadata(dht.metadata(), &data);
  SerializeHostBuffer(*dht.buffer(), &data);
  return data;
}

llvm::Expected<DenseHostTensor> DeserializeDenseHostTensorFromDenseAttr(
    DenseAttr attr, HostContext* host) {
  DType dtype = ConvertBEFAttributeTypeToTensorDType(attr.dtype());
  TensorMetadata md(dtype, attr.shape());

  auto result_alloc = DenseHostTensor::CreateUninitialized(md, host);
  if (!result_alloc) {
    return MakeStringError("error creating DenseHostTensor");
  }

  auto& result_tensor = result_alloc.getValue();
  std::memcpy(result_tensor.data(), attr.elements(), attr.DataSizeInBytes());
  return std::move(result_tensor);
}

TensorMetadata CreateTensorMetadata(const DenseAttr& attr) {
  return CreateDenseView(attr).metadata();
}

DenseView CreateDenseView(const DenseAttr& attr) {
  auto dtype = ConvertBEFAttributeTypeToTensorDType(attr.dtype());
  return DenseView(dtype, attr.shape(), attr.elements());
}

}  // namespace tfrt
