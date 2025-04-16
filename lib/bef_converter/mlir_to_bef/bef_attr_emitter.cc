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

// Emit attirbutes to the BEF attribute section.

#include "bef_attr_emitter.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/traits.h"
#include "tfrt/core_runtime/opdefs/types.h"

namespace tfrt {

DType BefAttrEmitter::EncodeIntegerTypeAttribute(
    mlir::IntegerType integer_type) {
  if (integer_type.isUnsigned()) {
    switch (integer_type.getWidth()) {
      case 8:
        return DType::UI8;
      case 16:
        return DType::UI16;
      case 32:
        return DType::UI32;
      case 64:
        return DType::UI64;
    }
  } else {
    switch (integer_type.getWidth()) {
      case 1:
        return DType::I1;
      case 8:
        return DType::I8;
      case 16:
        return DType::I16;
      case 32:
        return DType::I32;
      case 64:
        return DType::I64;
    }
  }

  llvm_unreachable("unknown integer type width.");
}

DType BefAttrEmitter::EncodeFloatTypeAttribute(mlir::FloatType float_type) {
  if (float_type.isBF16()) return DType::BF16;
  if (float_type.isF16()) return DType::F16;
  if (float_type.isF32()) return DType::F32;
  if (float_type.isF64()) return DType::F64;

  llvm_unreachable("unknown float type width.");
}

DType BefAttrEmitter::EncodeComplexTypeAttribute(
    mlir::ComplexType complex_type) {
  auto element_type = complex_type.getElementType();

  if (element_type.isF32()) return DType::Complex64;
  if (element_type.isF64()) return DType::Complex128;

  llvm_unreachable("unknown complex type width.");
}

DType BefAttrEmitter::ConvertMlirTypeToDType(mlir::Type type) {
  if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return EncodeIntegerTypeAttribute(integer_type);
  }

  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(type)) {
    return EncodeFloatTypeAttribute(float_type);
  }

  if (auto string_type = mlir::dyn_cast<corert::StringType>(type)) {
    return DType::String;
  }

  if (auto resource_type = mlir::dyn_cast<corert::ResourceType>(type)) {
    return DType::Resource;
  }

  if (auto variant_type = mlir::dyn_cast<corert::VariantType>(type)) {
    return DType::Variant;
  }

  if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    return EncodeComplexTypeAttribute(complex_type);
  }

  if (auto quantized_type = mlir::dyn_cast<corert::Quint8Type>(type)) {
    return DType::QUI8;
  }

  if (auto quantized_type = mlir::dyn_cast<corert::Quint16Type>(type)) {
    return DType::QUI16;
  }

  if (auto quantized_type = mlir::dyn_cast<corert::Qint8Type>(type)) {
    return DType::QI8;
  }

  if (auto quantized_type = mlir::dyn_cast<corert::Qint16Type>(type)) {
    return DType::QI16;
  }

  if (auto quantized_type = mlir::dyn_cast<corert::Qint32Type>(type)) {
    return DType::QI32;
  }

  llvm_unreachable("unknown type attribute");
}

// Return the kind of this attribute. If it is an array attribute, elements of
// it are checked recursively, and if any element is unsupported,
// BEFAttributeType::Unsupported will be returned.
BEFAttributeType BefAttrEmitter::GetBefAttributeType(mlir::Attribute attr) {
  // We support 1-bit (stored as 1 byte in BEF), 32-bit, and 64-bit
  // integers.
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    auto int_type = mlir::cast<mlir::IntegerType>(int_attr.getType());
    if (int_type.isUnsigned()) {
      switch (int_type.getWidth()) {
        case 8:
          return static_cast<BEFAttributeType>(DType::UI8);
        case 16:
          return static_cast<BEFAttributeType>(DType::UI16);
        case 32:
          return static_cast<BEFAttributeType>(DType::UI32);
        case 64:
          return static_cast<BEFAttributeType>(DType::UI64);
      }
    } else {
      switch (int_type.getWidth()) {
        case 1:
          return static_cast<BEFAttributeType>(DType::I1);
        case 8:
          return static_cast<BEFAttributeType>(DType::I8);
        case 16:
          return static_cast<BEFAttributeType>(DType::I16);
        case 32:
          return static_cast<BEFAttributeType>(DType::I32);
        case 64:
          return static_cast<BEFAttributeType>(DType::I64);
      }
    }
  }

  // We support BF16, F16, F32 and F64 floats.
  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    if (float_attr.getType().isBF16())
      return static_cast<BEFAttributeType>(DType::BF16);
    if (float_attr.getType().isF16())
      return static_cast<BEFAttributeType>(DType::F16);
    if (float_attr.getType().isF32())
      return static_cast<BEFAttributeType>(DType::F32);
    if (float_attr.getType().isF64())
      return static_cast<BEFAttributeType>(DType::F64);
  }

  // We support string attributes.
  if (mlir::isa<mlir::StringAttr>(attr))
    return static_cast<BEFAttributeType>(DType::String);

  // We support i1, i8, i16, i32, i64, ui8, ui16, ui32, ui64, bf16, f16, f32,
  //  f64, quint8, quint16, qint8, qint16, qint32, complex64, complex128,
  //  string, resource and variant type attributes.
  if (auto type_attr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    auto type = type_attr.getValue();
    if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
        type.isInteger(32) || type.isInteger(64) || type.isBF16() ||
        type.isF16() || type.isF32() || type.isF64() ||
        mlir::isa<corert::StringType>(type) ||
        mlir::isa<corert::ResourceType>(type) ||
        mlir::isa<corert::VariantType>(type) ||
        mlir::isa<corert::Quint8Type>(type) ||
        mlir::isa<corert::Quint16Type>(type) ||
        mlir::isa<corert::Qint8Type>(type) ||
        mlir::isa<corert::Qint16Type>(type) ||
        mlir::isa<corert::Qint32Type>(type))
      return BEFAttributeType::kType;

    if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
      auto element_type = complex_type.getElementType();
      if (element_type.isF32() || element_type.isF64())
        return BEFAttributeType::kType;
    }
  }

  // We support corert.shape attributes
  if (mlir::isa<tfrt::corert::ShapeAttr>(attr)) {
    return BEFAttributeType::kShape;
  }

  // We support dense attributes.
  if (auto dense_elements_attr =
          mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    auto element_type =
        ConvertMlirTypeToDType(dense_elements_attr.getType().getElementType());
    // We only support dense attributes with dtype element type. The exception
    // is that we don't support string dtype, because strings have variable
    // size.
    //
    // TODO(tfrt-devs): Consider supporting string elements in the dense
    // attribute.
    if (element_type == DType::UI8 || element_type == DType::UI16 ||
        element_type == DType::UI32 || element_type == DType::UI64 ||
        element_type == DType::I1 || element_type == DType::I8 ||
        element_type == DType::I16 || element_type == DType::I32 ||
        element_type == DType::I64 || element_type == DType::BF16 ||
        element_type == DType::F16 || element_type == DType::F32 ||
        element_type == DType::F64 || element_type == DType::Complex64 ||
        element_type == DType::Complex128)
      return BEFAttributeType::kDense;

    return BEFAttributeType::kUnsupported;
  }

  // We support arrays of supported attribute values.
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    if (array_attr.empty()) {
      return BEFAttributeType::kEmptyArray;
    }

    auto first_attr_type = GetBefAttributeType(*array_attr.begin());

    // Only fixed attributes can be included in an array.
    bool is_array = IsFixedAttribute(first_attr_type);

    for (auto elt : array_attr) {
      auto attr_type = GetBefAttributeType(elt);
      if (attr_type == BEFAttributeType::kUnsupported)
        return BEFAttributeType::kUnsupported;

      // Arrays requires all elements have the same type and the size.
      if (attr_type != first_attr_type) {
        is_array = false;
        break;
      }
    }

    if (is_array) return GetArrayAttributeType(first_attr_type);

    return BEFAttributeType::kAggregate;
  }

  // We support symbol references to compiled functions.
  if (auto symbol_ref_attr = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
    return BEFAttributeType::kSymbolRef;
  }

  return BEFAttributeType::kUnsupported;
}

static bool IsMatchedWithDType(BEFAttributeType attribute_type, DType dtype) {
  return (attribute_type == static_cast<BEFAttributeType>(dtype));
}

template <typename T>
size_t BefAttrEmitter::EmitIntegerAttribute(mlir::Attribute mlir_attr) {
  auto attr = mlir::cast<mlir::IntegerAttr>(mlir_attr);
  return EncodeAttr<T>(static_cast<T>(attr.getValue().getLimitedValue()));
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a kernel, even in recursive positions.
bool BefAttrEmitter::IsSupportedAttribute(mlir::Attribute attr) {
  // We support references to functions.
  if (mlir::isa<mlir::SymbolRefAttr>(attr)) return true;
  return GetBefAttributeType(attr) != BEFAttributeType::kUnsupported;
}

size_t BefAttrEmitter::EmitAttribute(BEFAttributeType attribute_type,
                                     mlir::Attribute mlir_attr) {
  if (IsMatchedWithDType(attribute_type, DType::UI8) ||
      IsMatchedWithDType(attribute_type, DType::I1))
    return EmitIntegerAttribute<uint8_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::UI16))
    return EmitIntegerAttribute<uint16_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::UI32))
    return EmitIntegerAttribute<uint32_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::UI64))
    return EmitIntegerAttribute<uint64_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::I8))
    return EmitIntegerAttribute<int8_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::I16))
    return EmitIntegerAttribute<int16_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::I32))
    return EmitIntegerAttribute<int32_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::I64))
    return EmitIntegerAttribute<int64_t>(mlir_attr);

  if (IsMatchedWithDType(attribute_type, DType::F32)) {
    auto attr = mlir::cast<mlir::FloatAttr>(mlir_attr);
    return EncodeAttr<float>(
        static_cast<float>(attr.getValue().convertToFloat()));
  }

  if (IsMatchedWithDType(attribute_type, DType::F64)) {
    auto attr = mlir::cast<mlir::FloatAttr>(mlir_attr);
    return EncodeAttr<double>(
        static_cast<double>(attr.getValue().convertToDouble()));
  }

  if (IsMatchedWithDType(attribute_type, DType::F16) ||
      IsMatchedWithDType(attribute_type, DType::BF16)) {
    auto attr = mlir::cast<mlir::FloatAttr>(mlir_attr);
    return EncodeAttr<uint16_t>(static_cast<uint16_t>(
        attr.getValue().bitcastToAPInt().getLimitedValue()));
  }

  if (IsMatchedWithDType(attribute_type, DType::String)) {
    auto attr = mlir::cast<mlir::StringAttr>(mlir_attr);
    return EncodeStringAttr(attr.getValue());
  }

  if (attribute_type == BEFAttributeType::kType) {
    auto attr = mlir::cast<mlir::TypeAttr>(mlir_attr);
    const auto dtype = ConvertMlirTypeToDType(attr.getValue());
    return EncodeAttr<uint8_t>(static_cast<uint8_t>(dtype));
  }

  if (attribute_type == BEFAttributeType::kShape) {
    tfrt::corert::ShapeAttr shape_attr =
        mlir::cast<tfrt::corert::ShapeAttr>(mlir_attr);
    return (shape_attr.hasRank()) ? EncodeRankedShapeAttr(shape_attr.getShape())
                                  : EncodeUnrankedShapeAttr();
  }

  if (IsArrayAttribute(attribute_type)) {
    return EmitArrayAttribute(attribute_type,
                              mlir::cast<mlir::ArrayAttr>(mlir_attr));
  }

  if (IsDenseAttribute(attribute_type)) {
    return EmitDenseAttribute(attribute_type,
                              mlir::cast<mlir::DenseElementsAttr>(mlir_attr));
  }

  if (attribute_type == BEFAttributeType::kAggregate) {
    return EmitAggregatedAttribute(mlir::cast<mlir::ArrayAttr>(mlir_attr));
  }

  llvm_unreachable("Unknown attribute");
}

size_t BefAttrEmitter::EmitArrayAttribute(BEFAttributeType attribute_type,
                                          mlir::ArrayAttr attr) {
  const auto element_count = attr.size();
  if (element_count == 0) {
    return EncodeEmptyAttr();
  }
  auto offset = EncodeArrayAttrHeader(
      element_count, GetAttributeDataTypeAlignment(attribute_type));
  for (auto element : attr) {
    EmitAttribute(element);
  }
  return offset;
}

size_t BefAttrEmitter::EmitDenseAttribute(BEFAttributeType attribute_type,
                                          mlir::DenseElementsAttr attr) {
  const auto shaped_type = attr.getType();
  assert(shaped_type.hasRank());
  const auto element_type =
      ConvertMlirTypeToDType(shaped_type.getElementType());

  const size_t element_count = shaped_type.getNumElements();

  const size_t offset =
      EncodeDenseAttrHeader(element_type, shaped_type.getShape(),
                            GetHostSize(element_type) * element_count);

  if (element_type == DType::I1) {
    for (bool bool_attr : attr.getValues<bool>()) {
      EmitByte(bool_attr ? 1 : 0);
    }
  } else {
    ArrayRef<char> raw_data = attr.getRawData();
    for (int i = 0; i < (attr.isSplat() ? element_count : 1); ++i) {
      EmitBytes(llvm::ArrayRef(
          reinterpret_cast<const uint8_t*>(raw_data.data()), raw_data.size()));
    }
  }

  return offset;
}

size_t BefAttrEmitter::EmitAggregatedAttribute(mlir::ArrayAttr attr) {
  const auto element_count = attr.size();
  if (element_count == 0) return EncodeEmptyAttr();

  size_t offset_offset;
  const size_t offset = EncodeAggregatedAttrHeader(
      GetMaximumAlignment(attr), element_count, &offset_offset);
  for (auto element : attr) {
    // Reserve a space for BEFAttributeType (1B)
    EmitByte(kDummyByte);

    EncodeAggregatedAttrEntryTypeAndOffset(offset, &offset_offset,
                                           GetBefAttributeType(element),
                                           EmitAttribute(element));
  }
  EncodeAggregatedAttrLength(offset);
  return offset;
}

size_t BefAttrEmitter::EmitSymbolRefAttribute(
    BefCompilationUnits& compilation_units, mlir::SymbolRefAttr attr) {
  const size_t offset = size();

  // Emit size information in VBR form for the SymbolRef and
  // serialized compilation unit.
  auto symbol = mlir::cast<mlir::SymbolRefAttr>(attr);

  // Emit the sequential id of the current symbol in the serialized module.
  size_t serialized_id = compilation_units.SerializedSymbolId(symbol);
  EmitVbrInt(serialized_id);

  // Length of the root symbol name.
  EmitVbrInt(symbol.getRootReference().getValue().size());

  // Lengths of the nested symbols names.
  size_t num_nested_refs = symbol.getNestedReferences().size();
  EmitVbrInt(num_nested_refs);
  llvm::SmallVector<size_t, 4> nested_ref_len(num_nested_refs);
  for (size_t i = 0; i < num_nested_refs; ++i)
    EmitVbrInt(symbol.getNestedReferences()[i].getValue().size());

  // Length of the serialized compilation unit.
  const auto sealized_size = compilation_units.SerializedOperationSize(symbol);
  EmitVbrInt(sealized_size);

  // Payload.
  EmitBytes(compilation_units.SerializedSymbolData(attr));
  EmitBytes(compilation_units.SerializedOperationData(attr));
  EmitByte(0);

  return offset;
}

size_t BefAttrEmitter::GetAlignment(mlir::Attribute attr) {
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return int_attr.getType().getIntOrFloatBitWidth() / 8;

  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return float_attr.getType().getIntOrFloatBitWidth() / 8;

  if (mlir::isa<mlir::StringAttr>(attr)) return alignof(AttrSizeT);

  if (mlir::isa<mlir::TypeAttr>(attr) || mlir::isa<mlir::SymbolRefAttr>(attr))
    return 1;

  if (mlir::isa<tfrt::corert::ShapeAttr>(attr)) {
    tfrt::corert::ShapeAttr shape_attr =
        mlir::cast<tfrt::corert::ShapeAttr>(attr);
    return (shape_attr.hasRank()) ? alignof(AttrShapeT) : 1;
  }

  if (auto dense_elements_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
    return kAttributeTensorAlignment;

  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(attr))
    return std::max(alignof(AttrShapeT), GetMaximumAlignment(array_attr));

  llvm_unreachable("Unknown attribute");
}

size_t BefAttrEmitter::GetMaximumAlignment(mlir::ArrayAttr attr) {
  size_t maximum = 1;
  for (auto element : attr) {
    maximum = std::max(maximum, GetAlignment(element));
  }
  return maximum;
}

}  // namespace tfrt
