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

// Read attirbutes from a BEF attribute section.

#ifndef TFRT_LIB_BEF_CONVERTER_BEF_TO_MLIR_BEF_ATTR_READER_H_
#define TFRT_LIB_BEF_CONVERTER_BEF_TO_MLIR_BEF_ATTR_READER_H_

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/host_context/attribute_utils.h"

namespace tfrt {

using CompilationUnits =
    llvm::SetVector<std::pair<mlir::Location, string_view>>;

// This class converts BEF attributes to mlir representations.
class BefAttrReader {
 public:
  BefAttrReader(ArrayRef<uint8_t> attributes, mlir::MLIRContext* context)
      : attributes_(attributes), context_(*context), builder_(context) {}

  // Read an attribute from the given offset.
  mlir::Attribute ReadAttribute(BEFAttributeType attribute_type, size_t offset);

  // Read a SymbolRefAttribute from the given offset.
  mlir::Attribute ReadSymbolRefAttribute(size_t offset, mlir::Location loc,
                                         CompilationUnits& compilation_units);

 private:
  mlir::Attribute ReadArrayAttribute(BEFAttributeType attribute_type,
                                     size_t offset);

  mlir::Attribute ReadDenseAttribute(size_t offset);

  mlir::Attribute ReadAggregateAttribute(size_t offset);

  ArrayRef<uint8_t> attributes_;
  mlir::MLIRContext& context_;
  mlir::Builder builder_;
};

}  // namespace tfrt

#endif
