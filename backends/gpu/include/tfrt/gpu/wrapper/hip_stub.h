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

// Mostly auto-generated HIP API header.
#ifndef TFRT_GPU_WRAPPER_HIP_STUB_H_
#define TFRT_GPU_WRAPPER_HIP_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"

// Define macros from hip_runtime.h used in hip_stub.h.inc.
#define __dparm(x)
#define DEPRECATED(x)

// Declare types from hip_runtime.h used in hip_stub.h.inc.
using hipDeviceptr_t = void*;
using dim3 = std::array<unsigned, 3>;
struct hipPointerAttribute_t;
struct hipFuncAttributes;
struct hipLaunchParams;

extern "C" {
#include "hip_stub.h.inc"
}

#undef __dparm
#undef DEPRECATED

const char* hipGetErrorName(hipError_t hip_error);
const char* hipGetErrorString(hipError_t hip_error);

// Enums for corresponding #defines in the HIP headers.
enum hipDeviceFlags_t {
  hipDeviceScheduleAuto = 0x0,
  hipDeviceScheduleSpin = 0x1,
  hipDeviceScheduleYield = 0x2,
  hipDeviceScheduleBlockingSync = 0x4,
  hipDeviceMapHost = 0x8,
  hipDeviceLmemResizeToMax = 0x10,
};
enum hipStreamFlags_t {
  hipStreamDefault = 0x0,
  hipStreamNonBlocking = 0x1,
};
enum hipEventFlags_t {
  hipEventDefault = 0x0,
  hipEventBlockingSync = 0x1,
  hipEventDisableTiming = 0x2,
  hipEventInterprocess = 0x4,
};
enum hipHostMallocFlags_t {
  hipHostMallocDefault = 0x0,
  hipHostMallocPortable = 0x1,
  hipHostMallocMapped = 0x2,
  hipHostMallocWriteCombined = 0x4,
  hipHostMallocCoherent = 0x40000000,
  hipHostMallocNonCoherent = 0x80000000,
};
enum hipHostRegisterFlags_t {
  hipHostRegisterDefault = 0x0,
  hipHostRegisterMapped = 0x2,
  hipExtHostRegisterCoarseGrained = 0x8,
};
enum hipMemAttachFlags_t {
  hipMemAttachGlobal = 1,
  hipMemAttachHost = 2,
};

// Attribute structs declared in HIP headers.
struct hipPointerAttribute_t {
  hipMemoryType memoryType;
  int device;
  void* devicePointer;
  void* hostPointer;
  int isManaged;
  unsigned allocationFlags;
};
struct hipFuncAttributes {
  int binaryVersion;
  int cacheModeCA;        // always 0.
  size_t constSizeBytes;  // always 0.
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;  // always 0.
  int ptxVersion;              // always 30.
  size_t sharedSizeBytes;
};

#endif  // TFRT_GPU_WRAPPER_HIP_STUB_H_
