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

// This file defines AlignedAlloc() for allocating dynamic buffer with
// explicit alignment.

#include "tfrt/support/alloc.h"

#include <cstdlib>

namespace tfrt {

void* AlignedAlloc(size_t alignment, size_t size) {
  if (alignment <= alignof(void*)) return std::malloc(size);
  size = (size + alignment - 1) / alignment * alignment;

#if defined(__ANDROID__) || defined(OS_ANDROID)
  return memalign(alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // alignof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  if (posix_memalign(&ptr, alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}

void AlignedFree(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace tfrt
