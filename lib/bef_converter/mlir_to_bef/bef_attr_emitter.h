/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// Emit attirbutes to a BEF attribute section.

#ifndef TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_ATTR_EMITTER_H_
#define TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_ATTR_EMITTER_H_

#include "bef_compilation_units.h"
#include "mlir/IR/BuiltinOps.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/core_runtime/opdefs/attributes.h"

namespace tfrt {

// Emit BEF attributes to a byte stream.
class BefAttrEmitter : public BefAttrEncoder {
 public:
  // Get DType from a mlir::Type instance.
  static DType ConvertMlirTypeToDType(mlir::Type type);

  // Get BEFAttributeType from a mlir::Attribute instance.
  static BEFAttributeType GetBefAttributeType(mlir::Attribute attr);

  // Check if an attribute is supported by BEF.
  static bool IsSupportedAttribute(mlir::Attribute attr);

  // Emit an attribute.
  size_t EmitAttribute(mlir::Attribute mlir_attr) {
    return EmitAttribute(GetBefAttributeType(mlir_attr), mlir_attr);
  }

  // Emit an attribute with a known attribute type.
  size_t EmitAttribute(BEFAttributeType attribute_type,
                       mlir::Attribute mlir_attr);

  // Emit a SymbolRefAttribute.
  size_t EmitSymbolRefAttribute(BefCompilationUnits& compilation_units,
                                mlir::SymbolRefAttr attr);

 private:
  template <typename T>
  size_t EmitIntegerAttribute(mlir::Attribute mlir_attr);

  static DType EncodeIntegerTypeAttribute(mlir::IntegerType integer_type);
  static DType EncodeFloatTypeAttribute(mlir::FloatType float_type);
  static DType EncodeComplexTypeAttribute(mlir::ComplexType complex_type);

  size_t EmitDenseAttribute(BEFAttributeType attribute_type,
                            mlir::DenseElementsAttr attr);
  size_t EmitArrayAttribute(BEFAttributeType attribute_type,
                            mlir::ArrayAttr attr);
  size_t EmitAggregatedAttribute(mlir::ArrayAttr attr);

  size_t GetAlignment(mlir::Attribute mlir_attr);
  size_t GetMaximumAlignment(mlir::ArrayAttr attr);
};

}  // namespace tfrt

#endif
