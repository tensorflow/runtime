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
namespace {

BEFDataType ConvertTensorDTypeToBEFDataType(DType dtype) {
  switch (dtype.kind()) {
    case DType::BOOL:
      return BEFDataType::kBool;
    case DType::UI8:
      return BEFDataType::kUI8;
    case DType::I8:
      return BEFDataType::kI8;
    case DType::I16:
      return BEFDataType::kI16;
    case DType::I32:
      return BEFDataType::kI32;
    case DType::I64:
      return BEFDataType::kI64;
    case DType::BF16:
      return BEFDataType::kBF16;
    case DType::F16:
      return BEFDataType::kF16;
    case DType::F32:
      return BEFDataType::kF32;
    case DType::F64:
      return BEFDataType::kF64;
    case DType::COMPLEX64:
      return BEFDataType::kComplex64;
    default:
      llvm_unreachable("unsupported dtype.");
  }
}

}  // namespace

DType ConvertBEFDataTypeToTensorDType(BEFDataType kind) {
  switch (kind) {
    case BEFDataType::kBool:
      return DType(DType::BOOL);
    case BEFDataType::kI8:
      return DType(DType::I8);
    case BEFDataType::kI16:
      return DType(DType::I16);
    case BEFDataType::kI32:
      return DType(DType::I32);
    case BEFDataType::kI64:
      return DType(DType::I64);
    case BEFDataType::kUI8:
      return DType(DType::UI8);
    case BEFDataType::kBF16:
      return DType(DType::BF16);
    case BEFDataType::kF16:
      return DType(DType::F16);
    case BEFDataType::kF32:
      return DType(DType::F32);
    case BEFDataType::kF64:
      return DType(DType::F64);
    case BEFDataType::kComplex64:
      return DType(DType::COMPLEX64);
    // TODO(tf-runtime-team): Support the missing dtypes in tensor.
    case BEFDataType::kUI16:
    case BEFDataType::kUI32:
    case BEFDataType::kUI64:
    case BEFDataType::kComplex128:
    default:
      llvm_unreachable("unsupported dtype.");
  }
}

// TODO(tfrt-devs): Consider creating a custom buffer with 8-byte
// alignment for the tensor data instead of using std::vector<uint64_t>.
std::vector<uint8_t> SerializeDenseHostTensorToDenseAttr(
    const DenseHostTensor& dht) {
  std::vector<uint8_t> data;

  const auto& md = dht.metadata();
  const auto& buf = *dht.buffer();

  BEFDenseAttr header;
  header.base.type =
      GetDenseAttributeType(ConvertTensorDTypeToBEFDataType(md.dtype));
  header.rank = md.shape.GetRank();
  header.num_elements = md.shape.GetNumElements();

  header.shape_offset = llvm::alignTo(sizeof(BEFDenseAttr), alignof(int64_t));
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
  header.base.byte_count = AssertAttrFieldSize(data.size());

  std::memcpy(data.data(), &header, sizeof(BEFDenseAttr));

  return data;
}

llvm::Expected<DenseHostTensor> DeserializeDenseHostTensorFromDenseAttr(
    DenseAttr attr, HostContext* host) {
  DType dtype = ConvertBEFDataTypeToTensorDType(attr.dtype());
  TensorMetadata md(dtype, attr.shape());

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
  auto dtype = ConvertBEFDataTypeToTensorDType(attr.dtype());
  return DenseView(dtype, attr.shape(), attr.GetElements());
}

}  // namespace tfrt
