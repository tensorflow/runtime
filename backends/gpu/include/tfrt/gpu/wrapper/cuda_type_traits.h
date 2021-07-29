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

// LLVM PointerLikeTypeTraits for half and complex types.
//
// Note: do not include this file at the same time as HIP headers because they
// define float2/double2 differently.

#ifndef TFRT_GPU_WRAPPER_CUDA_TYPE_TRAITS_H_
#define TFRT_GPU_WRAPPER_CUDA_TYPE_TRAITS_H_

// Forward declarations of half types.
struct __half;
struct __half2;

// Forward declaration of complex types.
using cuComplex = struct float2;
using cuDoubleComplex = struct double2;

namespace llvm {
// Define pointer traits for incomplete types.
template <typename>
struct PointerLikeTypeTraits;
template <>
struct PointerLikeTypeTraits<__half *> {
  static void *getAsVoidPointer(__half *ptr) { return ptr; }
  static __half *getFromVoidPointer(void *ptr) {
    return static_cast<__half *>(ptr);
  }
  // CUDA's __half (defined in vector_types.h) is aligned to 2 bytes.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 1;
};
template <>
struct PointerLikeTypeTraits<__half2 *> {
  static void *getAsVoidPointer(__half2 *ptr) { return ptr; }
  static __half2 *getFromVoidPointer(void *ptr) {
    return static_cast<__half2 *>(ptr);
  }
  // CUDA's __half2 (defined in vector_types.h) is aligned to 4 bytes.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 2;
};
template <>
struct PointerLikeTypeTraits<float2 *> {
  static void *getAsVoidPointer(float2 *ptr) { return ptr; }
  static float2 *getFromVoidPointer(void *ptr) {
    return static_cast<float2 *>(ptr);
  }
  // CUDA's float2 (defined in vector_types.h) is aligned to 8 bytes.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 3;
};
template <>
struct PointerLikeTypeTraits<double2 *> {
  static void *getAsVoidPointer(double2 *ptr) { return ptr; }
  static double2 *getFromVoidPointer(void *ptr) {
    return static_cast<double2 *>(ptr);
  }
  // CUDA's double2 (defined in vector_types.h) is aligned to 16 bytes.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 4;
};
}  // namespace llvm

#endif  // TFRT_GPU_WRAPPER_CUDA_TYPE_TRAITS_H_
