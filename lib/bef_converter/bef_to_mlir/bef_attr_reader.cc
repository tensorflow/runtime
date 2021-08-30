// Copyright 2021 The TensorFlow Runtime Authors
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

// Read attirbutes from a BEF attribute section.

#include "bef_attr_reader.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/traits.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {

static mlir::Type DecodeTypeAttribute(mlir::Builder* builder,
                                      DType attribute_type) {
  switch (attribute_type) {
    case DType::I1:
      return builder->getIntegerType(1);
    case DType::I8:
      return builder->getIntegerType(8);
    case DType::I16:
      return builder->getIntegerType(16);
    case DType::I32:
      return builder->getIntegerType(32);
    case DType::I64:
      return builder->getIntegerType(64);
    case DType::UI8:
      return builder->getIntegerType(8, /*isSigned=*/false);
    case DType::UI16:
      return builder->getIntegerType(16, /*isSigned=*/false);
    case DType::UI32:
      return builder->getIntegerType(32, /*isSigned=*/false);
    case DType::UI64:
      return builder->getIntegerType(64, /*isSigned=*/false);
    case DType::BF16:
      return builder->getBF16Type();
    case DType::F16:
      return builder->getF16Type();
    case DType::F32:
      return builder->getF32Type();
    case DType::F64:
      return builder->getF64Type();
    case DType::Complex64:
      return mlir::ComplexType::get(builder->getF32Type());
    case DType::Complex128:
      return mlir::ComplexType::get(builder->getF64Type());
    case DType::String:
      return tfrt::corert::StringType::get(builder->getContext());
    case DType::Resource:
      return tfrt::corert::ResourceType::get(builder->getContext());
    case DType::Variant:
      return tfrt::corert::VariantType::get(builder->getContext());
    case DType::QUI8:
      return tfrt::corert::Quint8Type::get(builder->getContext());
    case DType::QUI16:
      return tfrt::corert::Quint16Type::get(builder->getContext());
    case DType::QI8:
      return tfrt::corert::Qint8Type::get(builder->getContext());
    case DType::QI16:
      return tfrt::corert::Qint16Type::get(builder->getContext());
    case DType::QI32:
      return tfrt::corert::Qint32Type::get(builder->getContext());
    default:
      llvm_unreachable("unknown type attribute.");
  }
}

mlir::Attribute BefAttrReader::ReadAttribute(BEFAttributeType attribute_type,
                                             size_t offset) {
  if (IsArrayAttribute(attribute_type)) {
    return ReadArrayAttribute(attribute_type, offset);
  }

  if (IsDenseAttribute(attribute_type)) {
    return ReadDenseAttribute(offset);
  }

  if (attribute_type == BEFAttributeType::kAggregate) {
    return ReadAggregateAttribute(offset);
  }

  const auto ptr = &attributes_[offset];

  if (attribute_type == static_cast<BEFAttributeType>(DType::UI8)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 8, mlir::IntegerType::Unsigned),
        Attribute<uint8_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::UI16)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 16, mlir::IntegerType::Unsigned),
        Attribute<uint16_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::UI32)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 32, mlir::IntegerType::Unsigned),
        Attribute<uint32_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::UI64)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 64, mlir::IntegerType::Unsigned),
        Attribute<uint64_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::I1)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 1, mlir::IntegerType::Signless),
        Attribute<uint8_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::I8)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 8, mlir::IntegerType::Signless),
        Attribute<int8_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::I16)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 16, mlir::IntegerType::Signless),
        Attribute<int16_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::I32)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 32, mlir::IntegerType::Signless),
        Attribute<int32_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::I64)) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 64, mlir::IntegerType::Signless),
        Attribute<int64_t>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::F32)) {
    return mlir::FloatAttr::get(builder_.getF32Type(),
                                Attribute<float>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::F64)) {
    return mlir::FloatAttr::get(builder_.getF64Type(),
                                Attribute<double>(ptr).get());
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::F16) ||
      attribute_type == static_cast<BEFAttributeType>(DType::BF16)) {
    auto ftype = (attribute_type == static_cast<BEFAttributeType>(DType::F16))
                     ? builder_.getF16Type()
                     : builder_.getBF16Type();

    auto int_attr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 16, mlir::IntegerType::Unsigned),
        Attribute<uint16_t>(ptr).get());

    return mlir::FloatAttr::get(
        ftype, llvm::APFloat(ftype.getFloatSemantics(), int_attr.getValue()));
  }

  if (attribute_type == static_cast<BEFAttributeType>(DType::String)) {
    return mlir::StringAttr::get(&context_, StringAttribute(ptr).get());
  }

  if (attribute_type == BEFAttributeType::kType) {
    return mlir::TypeAttr::get(DecodeTypeAttribute(
        &builder_, static_cast<DType>(Attribute<uint8_t>(ptr).get())));
  }

  if (attribute_type == BEFAttributeType::kShape) {
    auto shape = ShapeAttr(ptr);
    return shape.HasRank()
               ? tfrt::corert::ShapeAttr::get(builder_.getContext(),
                                              shape.GetShape())
               : tfrt::corert::ShapeAttr::get(builder_.getContext());
  }

  llvm_unreachable("Unknown attribute");
}

mlir::Attribute BefAttrReader::ReadArrayAttribute(
    BEFAttributeType attribute_type, size_t offset) {
  const auto element_size = GetAttributeDataTypeByteSize(attribute_type);
  const auto element_type = GetElementAttributeType(attribute_type);
  size_t element_count;

  element_count = *(reinterpret_cast<const AttrSizeT*>(&attributes_[offset]));
  offset += sizeof(AttrSizeT);

  SmallVector<mlir::Attribute, 8> elements;
  elements.reserve(element_count);

  for (int idx = 0; idx < element_count; ++idx) {
    elements.push_back(ReadAttribute(element_type, offset));
    offset += element_size;
  }
  return mlir::ArrayAttr::get(&context_, elements);
}

mlir::Attribute BefAttrReader::ReadDenseAttribute(size_t offset) {
  auto attr = DenseAttr(&attributes_[offset]);

  const size_t element_count = attr.GetNumElements();
  const auto element_type = attr.dtype();
  const auto element_size = GetHostSize(element_type);
  auto type = mlir::RankedTensorType::get(
      attr.shape(), DecodeTypeAttribute(&builder_, element_type));

  if (element_type == DType::Complex64 || element_type == DType::Complex128) {
    auto raw_data = attr.GetRawData();
    bool is_splat = false;
    bool is_valid =
        mlir::DenseElementsAttr::isValidRawBuffer(type, raw_data, is_splat);
    // TODO(tfrt-dev): Improve the error reporting in bef_to_mlir.
    assert(is_valid);
    (void)is_valid;
    return mlir::DenseElementsAttr::getFromRawBuffer(type, raw_data, is_splat);
  }

  SmallVector<mlir::Attribute, 8> elements;
  elements.reserve(element_count);
  offset += attr.GetPrefixSize();
  for (int idx = 0; idx < element_count; ++idx) {
    elements.push_back(
        ReadAttribute(static_cast<BEFAttributeType>(element_type), offset));
    offset += element_size;
  }

  return mlir::DenseElementsAttr::get(type, elements);
}

mlir::Attribute BefAttrReader::ReadAggregateAttribute(size_t offset) {
  auto attr = AggregateAttr(&attributes_[offset]);
  const size_t element_count = attr.GetNumElements();
  if (element_count == 0) return mlir::ArrayAttr::get(&context_, {});

  SmallVector<mlir::Attribute, 8> elements;
  elements.reserve(element_count);
  for (int idx = 0; idx < element_count; ++idx) {
    elements.push_back(ReadAttribute(attr.GetElementType(idx),
                                     offset + attr.GetElementOffset(idx)));
  }
  return mlir::ArrayAttr::get(&context_, elements);
}

mlir::Attribute BefAttrReader::ReadSymbolRefAttribute(
    size_t offset, mlir::Location loc, CompilationUnits& compilation_units) {
  auto attr = CompilationUnitAttribute(&attributes_[offset]);

  auto nest_symbols = attr.nested_symbols();

  llvm::SmallVector<mlir::FlatSymbolRefAttr, 4> nested;
  for (auto nested_symbol : nest_symbols) {
    nested.push_back(mlir::FlatSymbolRefAttr::get(&context_, nested_symbol));
  }
  compilation_units.insert({loc, attr.serialized_operation()});
  return mlir::SymbolRefAttr::get(
      mlir::StringAttr::get(&context_, attr.root_symbol()), nested);
}

}  // namespace tfrt
