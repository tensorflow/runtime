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

// Defines llvm::DenseMapInfo specializations for tfrt::gpu::wrapper::Resource.

#ifndef TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_
#define TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_

#include "llvm/ADT/DenseMapInfo.h"
#include "tfrt/gpu/wrapper/wrapper.h"

namespace llvm {

template <typename CudaT, typename HipT>
struct DenseMapInfo<tfrt::gpu::wrapper::Resource<CudaT, HipT>> {
  using Resource = tfrt::gpu::wrapper::Resource<CudaT, HipT>;
  static inline Resource getEmptyKey() {
    return Resource(DenseMapInfo<void*>::getEmptyKey());
  }
  static inline Resource getTombstoneKey() {
    return Resource(DenseMapInfo<void*>::getTombstoneKey());
  }
  static unsigned getHashValue(const Resource& resource) {
    return static_cast<unsigned>(hash(resource));
  }
  static bool isEqual(const Resource& lhs, const Resource& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // TFRT_GPU_WRAPPER_DENSE_MAP_UTILS_H_
