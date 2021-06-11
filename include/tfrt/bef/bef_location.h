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

// Classes to read BEF locations.

#ifndef TFRT_BEF_BEF_LOCATION_H_
#define TFRT_BEF_BEF_LOCATION_H_

#include "llvm/ADT/SmallVector.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

// BEF location types
enum class BefLocationType : uint8_t {
  kUnknown = 0,
  kFileLineCol = 1,
  kName = 2,
  kCallSite = 3,
  kFused = 4,
};

// A base class for BEF locations.
class BefLocation {
 public:
  explicit BefLocation(const void *base)
      : base_(static_cast<const uint8_t *>(base)) {}

  const uint8_t *base() const { return base_; }
  BefLocationType type() const { return static_cast<BefLocationType>(*base_); }
  size_t length() const { return length_; }

  template <typename T>
  bool isa() const {
    return T::classof(*this);
  }
  template <typename T>
  T dyn_cast() const {
    return isa<T>() ? T(base_) : T(nullptr);
  }
  template <typename T>
  T cast() const {
    assert(isa<T>());
    return T(base_);
  }

  explicit operator bool() const { return base_ != nullptr; }

 protected:
  string_view OffsetToString(ArrayRef<uint8_t> location_strings_section,
                             size_t offset) const;

  static const uint8_t *NextLocation(const uint8_t *ptr);

  const uint8_t *base_ = nullptr;
  size_t length_ = 0;
};

class BefUnknownLocation : public BefLocation {
 public:
  explicit BefUnknownLocation(const void *base);
  static bool classof(BefLocation base) {
    return base.type() == BefLocationType::kUnknown;
  }
};

class BefFileLineColLocation : public BefLocation {
 public:
  explicit BefFileLineColLocation(const void *base);

  string_view filename(ArrayRef<uint8_t> location_strings_section) const {
    return OffsetToString(location_strings_section, filename_offset_);
  }
  size_t line() const { return line_; }
  size_t column() const { return column_; }

  static bool classof(BefLocation base) {
    return base.type() == BefLocationType::kFileLineCol;
  }

 protected:
  size_t filename_offset_ = 0;
  size_t line_ = 0;
  size_t column_ = 0;
};

class BefNameLocation : public BefLocation {
 public:
  explicit BefNameLocation(const void *base);

  string_view name(ArrayRef<uint8_t> location_strings_section) const {
    return OffsetToString(location_strings_section, name_offset_);
  }

  BefLocation child() const { return BefLocation(child_); }

  static bool classof(BefLocation base) {
    return base.type() == BefLocationType::kName;
  }

 private:
  size_t name_offset_ = 0;
  const uint8_t *child_ = nullptr;
};

class BefCallSiteLocation : public BefLocation {
 public:
  explicit BefCallSiteLocation(const void *base);

  BefLocation callee() const { return BefLocation(callee_); }

  BefLocation caller() const { return BefLocation(caller_); }

  static bool classof(BefLocation base) {
    return base.type() == BefLocationType::kCallSite;
  }

 private:
  const uint8_t *callee_ = nullptr;
  const uint8_t *caller_ = nullptr;
};

class BefFusedLocation : public BefLocation {
 public:
  explicit BefFusedLocation(const void *base);

  size_t size() const { return size_; }

  BefLocation GetLocation(int index) const {
    assert(index < size_);
    return BefLocation(bases_[index]);
  }

  static bool classof(BefLocation base) {
    return base.type() == BefLocationType::kFused;
  }

 private:
  size_t size_;
  SmallVector<const uint8_t *, 4> bases_;
};

}  // namespace tfrt

#endif  // TFRT_BEF_BEF_LOCATION_H_
