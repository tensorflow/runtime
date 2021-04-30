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

// MIOpen enum parsers and printers.
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/miopen_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return os << "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return os << "miopenStatusNotInitialized";
    case miopenStatusInvalidValue:
      return os << "miopenStatusInvalidValue";
    case miopenStatusBadParm:
      return os << "miopenStatusBadParm";
    case miopenStatusAllocFailed:
      return os << "miopenStatusAllocFailed";
    case miopenStatusInternalError:
      return os << "miopenStatusInternalError";
    case miopenStatusNotImplemented:
      return os << "miopenStatusNotImplemented";
    case miopenStatusUnknownError:
      return os << "miopenStatusUnknownError";
    case miopenStatusUnsupportedOp:
      return os << "miopenStatusUnsupportedOp";
    default:
      return os << llvm::formatv("miopenStatus_t({0})",
                                 static_cast<int>(status));
  }
}

template <>
Expected<miopenDataType_t> Parse<miopenDataType_t>(llvm::StringRef name) {
  if (name == "miopenHalf") return miopenHalf;
  if (name == "miopenFloat") return miopenFloat;
  if (name == "miopenInt32") return miopenInt32;
  if (name == "miopenInt8") return miopenInt8;
  if (name == "miopenInt8x4") return miopenInt8x4;
  if (name == "miopenBFloat16") return miopenBFloat16;
  return MakeStringError("Unknown miopenDataType_t: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, miopenDataType_t value) {
  switch (value) {
    case miopenHalf:
      return os << "miopenHalf";
    case miopenFloat:
      return os << "miopenFloat";
    case miopenInt32:
      return os << "miopenInt32";
    case miopenInt8:
      return os << "miopenInt8";
    case miopenInt8x4:
      return os << "miopenInt8x4";
    case miopenBFloat16:
      return os << "miopenBFloat16";
    default:
      return os << llvm::formatv("miopenDataType_t({0})",
                                 static_cast<int>(value));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
