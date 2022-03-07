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

// Mostly auto-generated rocsolver API header.
#ifndef TFRT_GPU_WRAPPER_ROCSOLVER_STUB_H_
#define TFRT_GPU_WRAPPER_ROCSOLVER_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"
#include "tfrt/gpu/wrapper/rocblas_stub.h"

#ifdef _WIN32
#define ROCSOLVER_EXPORT __declspec(dllexport)
#else
#define ROCSOLVER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
#include "rocsolver_stub.h.inc"
}

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_STUB_H_
