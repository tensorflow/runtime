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

// Emit a mlir::Location as a BefLocation.

#ifndef TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_LOCATION_EMITTER_H_
#define TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_LOCATION_EMITTER_H_

#include "bef_string_emitter.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// Emit a location.
class BefLocationEmitter : public BefEmitter {
 public:
  BefLocationEmitter()
      : virtual_location_offset_(std::numeric_limits<uint32_t>::max()),
        emitted_location_count_(0) {}

  // Check if the location is convertible as a BefLocation.
  static bool IsSupportedLocation(const mlir::Location& loc);

  // Emit a location.
  size_t EmitLocation(const mlir::Location& loc);

  // Emit a location of an op.
  size_t EmitOpLocation(mlir::Operation* op);

  // Get the strings section data.
  ArrayRef<uint8_t> GetStringsSection() const;

  // Get the strings section emitter.
  BefEmitter& GetStringsSectionEmitter() { return strings_; }

  // Get the number of non-virtual emitted locations.
  size_t GetConcreteLocationCount() const { return emitted_location_count_; }

 private:
  static size_t CountSupportedLocations(const mlir::FusedLoc& loc);
  void EmitLocationStringAsVbrOffset(string_view str);

  BefStringEmitter strings_;
  uint32_t virtual_location_offset_;
  size_t emitted_location_count_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_CONVERTER_MLIR_TO_BEF_BEF_LOCATION_EMITTER_H_
