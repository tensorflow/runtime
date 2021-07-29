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

// NCCL enum parsers and printers.
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return os << "ncclSuccess";
    case ncclUnhandledCudaError:
      return os << "ncclUnhandledCudaError";
    case ncclSystemError:
      return os << "ncclSystemError";
    case ncclInternalError:
      return os << "ncclInternalError";
    case ncclInvalidArgument:
      return os << "ncclInvalidArgument";
    case ncclInvalidUsage:
      return os << "ncclInvalidUsage";
    default:
      return os << llvm::formatv("ncclResult_t({0})", static_cast<int>(result));
  }
}

template <>
Expected<ncclDataType_t> Parse<ncclDataType_t>(llvm::StringRef name) {
  if (name == "ncclInt8") return ncclInt8;
  if (name == "ncclChar") return ncclChar;
  if (name == "ncclUint8") return ncclUint8;
  if (name == "ncclInt32") return ncclInt32;
  if (name == "ncclInt") return ncclInt;
  if (name == "ncclUint32") return ncclUint32;
  if (name == "ncclInt64") return ncclInt64;
  if (name == "ncclUint64") return ncclUint64;
  if (name == "ncclFloat16") return ncclFloat16;
  if (name == "ncclHalf") return ncclHalf;
  if (name == "ncclFloat32") return ncclFloat32;
  if (name == "ncclFloat") return ncclFloat;
  if (name == "ncclFloat64") return ncclFloat64;
  if (name == "ncclDouble") return ncclDouble;
  return MakeStringError("Unknown ncclDataType_t: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclDataType_t value) {
  switch (value) {
    case ncclInt8:
      return os << "ncclInt8";
    case ncclUint8:
      return os << "ncclUint8";
    case ncclInt32:
      return os << "ncclInt32";
    case ncclUint32:
      return os << "ncclUint32";
    case ncclInt64:
      return os << "ncclInt64";
    case ncclUint64:
      return os << "ncclUint64";
    case ncclFloat16:
      return os << "ncclFloat16";
    case ncclFloat32:
      return os << "ncclFloat32";
    case ncclFloat64:
      return os << "ncclFloat64";
    default:
      return os << llvm::formatv("ncclDataType_t({0})",
                                 static_cast<int>(value));
  }
}

template <>
Expected<ncclRedOp_t> Parse<ncclRedOp_t>(llvm::StringRef name) {
  if (name == "ncclSum") return ncclSum;
  if (name == "ncclProd") return ncclProd;
  if (name == "ncclMax") return ncclMax;
  if (name == "ncclMin") return ncclMin;
  if (name == "ncclNumOps") return ncclNumOps;
  return MakeStringError("Unknown ncclRedOp_t: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclRedOp_t value) {
  switch (value) {
    case ncclSum:
      return os << "ncclSum";
    case ncclProd:
      return os << "ncclProd";
    case ncclMax:
      return os << "ncclMax";
    case ncclMin:
      return os << "ncclMin";
    case ncclNumOps:
      return os << "ncclNumOps";
    default:
      return os << llvm::formatv("ncclRedOp_t({0})", static_cast<int>(value));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
