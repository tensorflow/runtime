/*
 * Copyright 2021 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicablae law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares some utilities to access and distribute debug info.
#ifndef TFRT_HOST_CONTEXT_DEBUG_INFO_H_
#define TFRT_HOST_CONTEXT_DEBUG_INFO_H_

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace tfrt {
using DebugInfoEntry = llvm::StringRef;
using DebugInfoOffset = uint32_t;
class BEFKernel;

// Decode DebugInfo given a kernel.
class DebugInfoDecoder {
 public:
  virtual llvm::Optional<DebugInfoEntry> DecodeDebugInfo(BEFKernel*) const = 0;
  virtual ~DebugInfoDecoder() = default;
};

// Encapsulate a decoder and an offset.
struct DebugInfo {
  DebugInfoDecoder* decoder = nullptr;
  BEFKernel* kernel = nullptr;

  llvm::Optional<DebugInfoEntry> GetDebugInfo() const {
    if (decoder && kernel) {
      return decoder->DecodeDebugInfo(kernel);
    } else {
      return llvm::None;
    }
  }
};
}  // namespace tfrt

#endif