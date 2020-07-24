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

//===- device_util.h --------------------------------------------*- C++ -*-===//
//
// This file declares GPU device util functions for managing GPU devices.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_DEVICE_DEVICE_UTIL_H_
#define TFRT_GPU_DEVICE_DEVICE_UTIL_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {
class HostContext;
class GpuDevice;

namespace gpu {

// Create and return a GPU device. If the device has been created before
// return the existing device directly. Thread-safe.
llvm::Expected<RCReference<GpuDevice>> GetOrCreateGpuDevice(int gpu_ordinal,
                                                            HostContext* host);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_DEVICE_DEVICE_UTIL_H_
