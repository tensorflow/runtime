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

//===- hash_utils.h - Declares std::hash specializations --------*- C++ -*-===//
//
// Declares std::hash specializations for some tfrt::gpu::stream types.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_STREAM_HASH_UTILS_H_
#define TFRT_GPU_STREAM_HASH_UTILS_H_

#include <functional>

#include "tfrt/gpu/stream/stream_wrapper.h"

namespace std {
template <>
struct hash<tfrt::gpu::stream::Device> {
  size_t operator()(const tfrt::gpu::stream::Device& device) const noexcept;
};

template <>
struct hash<tfrt::gpu::stream::Stream> {
  size_t operator()(const tfrt::gpu::stream::Stream& stream) const noexcept;
};
}  // namespace std

#endif  // TFRT_GPU_STREAM_HASH_UTILS_H_
