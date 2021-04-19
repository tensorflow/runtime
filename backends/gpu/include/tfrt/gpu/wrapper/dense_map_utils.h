/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// Defines DenseMapInfo specializations
//
// Defines specializations for llvm::DenseMapInfo. This allows some gpu::stream
// classes to be used as keys in llvm::DenseMap.
#ifndef TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_
#define TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_

#include "llvm/ADT/DenseMapInfo.h"
#include "tfrt/gpu/wrapper/hash_utils.h"
#include "tfrt/gpu/wrapper/wrapper.h"

namespace llvm {

template <>
struct DenseMapInfo<tfrt::gpu::wrapper::Stream> {
  using Stream = tfrt::gpu::wrapper::Stream;
  static inline Stream getEmptyKey() {
    return Stream(DenseMapInfo<void*>::getEmptyKey());
  }
  static inline Stream getTombstoneKey() {
    return Stream(DenseMapInfo<void*>::getTombstoneKey());
  }
  static unsigned getHashValue(const Stream& stream) {
    return std::hash<Stream>{}(stream);
  }
  static bool isEqual(const Stream& lhs, const Stream& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_
