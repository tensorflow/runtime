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

// This file defines DTypes supported by Eigen.
#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_DTYPE_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_DTYPE_H_

#define EIGEN_USE_THREADS

#include "tfrt/dtype/dtype.h"
#include "tfrt/support/fp16.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tfrt {

template <DType::Kind K>
using EigenTypeForDTypeKind =
    std::conditional_t<std::is_same<fp16, TypeForDTypeKind<K>>::value,
                       Eigen::half, TypeForDTypeKind<K>>;
TFRT_REGISTER_DTYPE(Eigen::half, F16)
}  // namespace tfrt

namespace llvm {
template <typename>
struct PointerLikeTypeTraits;
template <>
struct PointerLikeTypeTraits<Eigen::half *> {
  static inline void *getAsVoidPointer(Eigen::half *ptr) { return ptr; }
  static inline Eigen::half *getFromVoidPointer(void *ptr) {
    return static_cast<Eigen::half *>(ptr);
  }
  // alignof(Eigen::half) == 2 (defined in Eigen/src/Core/arch/Default/Half.h).
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 2;
};
}  // namespace llvm

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_DTYPE_H_
