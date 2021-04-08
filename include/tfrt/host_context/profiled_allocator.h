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

// Profiled Memory Allocator
//
// This file implements a profiling host memory allocator that does a memory
// leak check and prints allocation statistics when destroyed.

#include <memory>

#include "tfrt/host_context/host_allocator.h"

namespace tfrt {

// Decorate an allocator with memory usage profiling.
std::unique_ptr<HostAllocator> CreateProfiledAllocator(
    std::unique_ptr<HostAllocator> allocator);

// Decorate an allocator with memory leak check.
std::unique_ptr<HostAllocator> CreateLeakCheckAllocator(
    std::unique_ptr<HostAllocator> allocator);

}  // namespace tfrt
