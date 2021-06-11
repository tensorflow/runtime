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

// BefLocationEmitter class to emit Locations section.

#include "bef_location_emitter.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "tfrt/bef/bef_location.h"

namespace tfrt {

bool BefLocationEmitter::IsSupportedLocation(const mlir::Location& loc) {
  if (loc.isa<mlir::UnknownLoc>()) return true;
  if (loc.isa<mlir::NameLoc>()) return true;
  if (loc.isa<mlir::FileLineColLoc>()) return true;
  if (auto callsite_loc = loc.dyn_cast<mlir::CallSiteLoc>()) {
    return IsSupportedLocation(callsite_loc.getCallee()) &&
           IsSupportedLocation(callsite_loc.getCaller());
  }
  if (auto fused_loc = loc.dyn_cast<mlir::FusedLoc>()) {
    for (auto& location : fused_loc.getLocations()) {
      if (IsSupportedLocation(location)) return true;
    }
  }
  return false;
}

size_t BefLocationEmitter::CountSupportedLocations(const mlir::FusedLoc& loc) {
  size_t supported_location_count = 0;
  for (auto& location : loc.getLocations()) {
    if (IsSupportedLocation(location)) ++supported_location_count;
  }
  return supported_location_count;
}

ArrayRef<uint8_t> BefLocationEmitter::GetStringsSection() const {
  auto& result = strings_.result();
  return llvm::makeArrayRef(result.data(), result.size());
}

void BefLocationEmitter::EmitLocationStringAsVbrOffset(string_view str) {
  EmitVbrInt(strings_.EmitString(str));
}

size_t BefLocationEmitter::EmitLocation(const mlir::Location& loc) {
  assert(IsSupportedLocation(loc));
  const size_t offset = size();

  if (auto filelinecol_loc = loc.dyn_cast<mlir::UnknownLoc>()) {
    // Encoding format: `0x00`
    EmitByte(static_cast<uint8_t>(BefLocationType::kUnknown));
    return offset;
  }

  if (auto filelinecol_loc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    // Encoding format: `0x01` FILELINECOL_LOC ::= `0x00` INDEX<"Filename">
    //                   INTEGER<"LineNum"> INTEGER<"ColumnNum">
    EmitByte(static_cast<uint8_t>(BefLocationType::kFileLineCol));
    EmitLocationStringAsVbrOffset(filelinecol_loc.getFilename());
    EmitVbrInt(filelinecol_loc.getLine());
    EmitVbrInt(filelinecol_loc.getColumn());
    return offset;
  }

  if (auto name_loc = loc.dyn_cast<mlir::NameLoc>()) {
    // Encoding format: `0x02` INDEX<"Name"> LOCATION<"Child">
    EmitByte(static_cast<uint8_t>(BefLocationType::kName));
    EmitLocationStringAsVbrOffset(name_loc.getName());
    EmitLocation(name_loc.getChildLoc());
    return offset;
  }

  if (auto callsite_loc = loc.dyn_cast<mlir::CallSiteLoc>()) {
    // Encoding format: `x003` LOCATION<"Callee"> LOCATION<"Caller">
    EmitByte(static_cast<uint8_t>(BefLocationType::kCallSite));
    EmitLocation(callsite_loc.getCallee());
    EmitLocation(callsite_loc.getCaller());
    return offset;
  }

  if (auto fused_loc = loc.dyn_cast<mlir::FusedLoc>()) {
    const size_t location_count = CountSupportedLocations(fused_loc);
    if (location_count > 0) {
      // Encoding format: `0x04` INTEGER<”NumLocations”> LOCATION*
      EmitByte(static_cast<uint8_t>(BefLocationType::kFused));
      EmitVbrInt(location_count);
      for (auto& location : fused_loc.getLocations()) {
        if (IsSupportedLocation(location)) EmitLocation(location);
      }
    }
    return offset;
  }

  llvm_unreachable("Unexpected location found.");
}

size_t BefLocationEmitter::EmitOpLocation(mlir::Operation* op) {
  auto loc = op->getLoc();

  if (!IsSupportedLocation(loc)) {
    return --virtual_location_offset_;
  }

  ++emitted_location_count_;

  size_t offset = EmitLocation(loc);
  assert(offset < virtual_location_offset_);

  return offset;
}

}  // namespace tfrt
