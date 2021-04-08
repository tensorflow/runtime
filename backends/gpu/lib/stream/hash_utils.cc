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

// Defines std::hash specializations
//
// Defines std::hash specializations for some tfrt::gpu::stream types.

#include "tfrt/gpu/stream/hash_utils.h"

// Algorithm of boost::hash_combine
static size_t HashCombine(size_t x, size_t y) {
  return x + 0x9e3779b9 + (y << 6) + (y >> 2);
}

namespace std {

size_t hash<tfrt::gpu::stream::Device>::operator()(
    const tfrt::gpu::stream::Device& device) const noexcept {
  auto platform = device.platform();
  auto id = device.id(platform);
  return HashCombine(std::hash<decltype(id)>{}(id),
                     std::hash<decltype(platform)>{}(platform));
}

size_t hash<tfrt::gpu::stream::Stream>::operator()(
    const tfrt::gpu::stream::Stream& stream) const noexcept {
  return std::hash<void*>{}(stream.pair_.getOpaqueValue());
}

}  // namespace std
