/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// Thin abstraction layer for CUDA and HIP runtime API.
//
// Note: prefer the driver API (see driver_wrapper.h) over this runtime API.
#ifndef TFRT_GPU_WRAPPER_RUNTIME_WRAPPER_H_
#define TFRT_GPU_WRAPPER_RUNTIME_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Error Free(std::nullptr_t, Platform platform);

llvm::Expected<int> RuntimeGetVersion(Platform platform);
llvm::Error GetLastError(CurrentContext current);
llvm::Error PeekAtLastError(CurrentContext current);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_RUNTIME_WRAPPER_H_
