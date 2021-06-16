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

#include "tfrt/bef/bef_location.h"

#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

string_view BefLocation::OffsetToString(
    ArrayRef<uint8_t> location_strings_section, size_t offset) const {
  assert(offset < location_strings_section.size());
  return string_view(
      reinterpret_cast<const char*>(location_strings_section.data() + offset));
}

const uint8_t* BefLocation::NextLocation(const uint8_t* ptr) {
  BefLocation loc(ptr);
  if (auto unknown = loc.dyn_cast<BefUnknownLocation>()) {
    return ptr + unknown.length();
  }

  if (auto filelinecol = loc.dyn_cast<BefFileLineColLocation>()) {
    return ptr + filelinecol.length();
  }

  if (auto name = loc.dyn_cast<BefNameLocation>()) {
    return ptr + name.length();
  }

  if (auto callsite = loc.dyn_cast<BefCallSiteLocation>()) {
    return ptr + callsite.length();
  }

  if (auto fused = loc.dyn_cast<BefFusedLocation>()) {
    return ptr + fused.length();
  }

  llvm_unreachable("Unexpected location found.");
}

BefUnknownLocation::BefUnknownLocation(const void* base) : BefLocation(base) {
  if (base) {
    assert(static_cast<BefLocationType>(*base_) == BefLocationType::kUnknown);
    length_ = 1;
  }
}

BefFileLineColLocation::BefFileLineColLocation(const void* base)
    : BefLocation(base) {
  if (base) {
    assert(static_cast<BefLocationType>(*base_) ==
           BefLocationType::kFileLineCol);
    auto ptr = ReadVbrInt(base_ + 1, &filename_offset_);
    ptr = ReadVbrInt(ptr, &line_);
    ptr = ReadVbrInt(ptr, &column_);
    length_ = ptr - base_;
  }
}

BefNameLocation::BefNameLocation(const void* base) : BefLocation(base) {
  if (base) {
    assert(static_cast<BefLocationType>(*base_) == BefLocationType::kName);
    auto ptr = ReadVbrInt(base_ + 1, &name_offset_);
    child_ = ptr;
    length_ = NextLocation(child_) - base_;
  }
}

BefCallSiteLocation::BefCallSiteLocation(const void* base) : BefLocation(base) {
  if (base) {
    assert(static_cast<BefLocationType>(*base_) == BefLocationType::kCallSite);
    callee_ = base_ + 1;
    caller_ = NextLocation(callee_);
    length_ = NextLocation(caller_) - base_;
  }
}

BefFusedLocation::BefFusedLocation(const void* base) : BefLocation(base) {
  if (base) {
    assert(static_cast<BefLocationType>(*base_) == BefLocationType::kFused);
    auto ptr = ReadVbrInt(base_ + 1, &size_);
    bases_.reserve(size_);
    for (auto idx = 0; idx < size_; idx++) {
      bases_.push_back(ptr);
      ptr = NextLocation(ptr);
    }
    length_ = ptr - base_;
  }
}

static std::string BefLocationToStr(ArrayRef<uint8_t> location_strings_section,
                                    const BefLocation& loc) {
  std::string out;
  llvm::raw_string_ostream decoded(out);
  if (auto filelinecol = loc.dyn_cast<BefUnknownLocation>()) {
    decoded << "(unknown)";
  } else if (auto filelinecol = loc.dyn_cast<BefFileLineColLocation>()) {
    decoded << filelinecol.filename(location_strings_section).str() << ";"
            << filelinecol.line() << ";" << filelinecol.column();
  } else if (auto name = loc.dyn_cast<BefNameLocation>()) {
    auto has_child = !name.child().isa<BefUnknownLocation>();
    if (has_child) {
      decoded << BefLocationToStr(location_strings_section, name.child());
      decoded << "(";
    }
    decoded << name.name(location_strings_section).str();
    if (has_child) decoded << ")";
  } else if (auto callsite = loc.dyn_cast<BefCallSiteLocation>()) {
    decoded << BefLocationToStr(location_strings_section, callsite.callee());
    decoded << "<-";
    decoded << BefLocationToStr(location_strings_section, callsite.caller());
  } else if (auto fused = loc.dyn_cast<BefFusedLocation>()) {
    for (int idx = 0; idx < fused.size(); ++idx) {
      if (idx > 0) decoded << ",";
      decoded << BefLocationToStr(location_strings_section,
                                  fused.GetLocation(idx));
    }
  } else {
    llvm_unreachable("Unexpected location found.");
  }
  return out;
}

DecodedLocation DecodeBefLocation(ArrayRef<uint8_t> location_strings_section,
                                  const BefLocation& loc) {
  DecodedLocation result;
  if (auto filelinecol = loc.dyn_cast<BefFileLineColLocation>()) {
    result = FileLineColLocation{
        filelinecol.filename(location_strings_section).str(),
        static_cast<int>(filelinecol.line()),
        static_cast<int>(filelinecol.column())};
  } else {
    result = OpaqueLocation{BefLocationToStr(location_strings_section, loc)};
  }
  return result;
}

Optional<DebugInfo> GetDebugInfoFromBefLocation(
    ArrayRef<uint8_t> location_strings_section, const BefLocation& loc) {
  // Treats BefNameLocation as DebugInfo.
  if (auto name = loc.dyn_cast<BefNameLocation>()) {
    return DebugInfo{name.name(location_strings_section).str()};
  }

  // Search a BefNameLocation.
  if (auto fused = loc.dyn_cast<BefFusedLocation>()) {
    for (int idx = 0; idx < fused.size(); ++idx) {
      auto child = fused.GetLocation(idx);
      if (auto name = child.dyn_cast<BefNameLocation>()) {
        return DebugInfo{name.name(location_strings_section).str()};
      }
    }
  }

  // Check if callee is a BefNameLocation.
  if (auto callsite = loc.dyn_cast<BefCallSiteLocation>()) {
    auto callee = callsite.callee();
    if (auto name = callee.dyn_cast<BefNameLocation>()) {
      return DebugInfo{name.name(location_strings_section).str()};
    }
  }

  return llvm::None;
}

}  // namespace tfrt
