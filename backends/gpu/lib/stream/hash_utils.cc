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

//===- hash_utils.cc - Defines std::hash specializations --------*- C++ -*-===//
//
// Defines std::hash specializations for some tfrt::gpu::stream types.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/stream/hash_utils.h"

namespace std {

size_t hash<tfrt::gpu::stream::Device>::operator()(
    const tfrt::gpu::stream::Device& x) const noexcept {
  size_t v = std::hash<tfrt::gpu::stream::Platform>()(x.platform());
  // Algorithm of boost::hash_combine
  return std::hash<int>()(x.id(x.platform())) + 0x9e3779b9 + (v << 6) +
         (v >> 2);
}

size_t hash<tfrt::gpu::stream::Stream>::operator()(
    const tfrt::gpu::stream::Stream& x) const noexcept {
  return x.hash();
}

}  // namespace std
