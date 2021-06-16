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

// Read a location from a BEF locations section.

#include "bef_location_reader.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "tfrt/bef/bef_location.h"

namespace tfrt {

mlir::Location BefLocationReader::ReadLocation(BefLocation loc) {
  if (auto unknown = loc.dyn_cast<BefUnknownLocation>()) {
    return mlir::UnknownLoc::get(&context_);
  } else if (auto filelinecol = loc.dyn_cast<BefFileLineColLocation>()) {
    return mlir::FileLineColLoc::get(&context_,
                                     filelinecol.filename(location_strings_),
                                     filelinecol.line(), filelinecol.column());
  } else if (auto name = loc.dyn_cast<BefNameLocation>()) {
    auto identifier =
        mlir::Identifier::get(name.name(location_strings_), &context_);
    auto child = ReadLocation(name.child());
    return mlir::NameLoc::get(identifier, child);
  } else if (auto callsite = loc.dyn_cast<BefCallSiteLocation>()) {
    auto callee = ReadLocation(callsite.callee().base() - locations_.data());
    auto caller = ReadLocation(callsite.caller().base() - locations_.data());
    return mlir::CallSiteLoc::get(callee, caller);
  } else if (auto fused = loc.dyn_cast<BefFusedLocation>()) {
    llvm::SmallVector<mlir::Location, 3> locations;
    for (int idx = 0; idx < fused.size(); ++idx) {
      locations.push_back(ReadLocation(fused.GetLocation(idx)));
    }
    return mlir::FusedLoc::get(&context_, locations);
  }

  llvm_unreachable("Unexpected location found.");
}

mlir::Location BefLocationReader::ReadLocation(size_t offset) {
  return ReadLocation(BefLocation(locations_.data() + offset));
}

}  // namespace tfrt
