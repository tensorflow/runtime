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

// Mostly auto-generated hipfft API header.
#ifndef TFRT_GPU_WRAPPER_HIPFFT_STUB_H_
#define TFRT_GPU_WRAPPER_HIPFFT_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"

#ifdef _WIN32
#define HIPFFT_EXPORT __declspec(dllexport)
#else
#define HIPFFT_EXPORT __attribute__((visibility("default")))
#endif

#define HIPFFT_FORWARD -1

#define HIPFFT_BACKWARD 1

// Declare types from hipfft.h used in hipfft_stub.h.inc.
struct hipfftComplex {
  float x, y;
};
struct hipfftDoubleComplex {
  double x, y;
};
using hipfftReal = float;
using hipfftDoubleReal = double;

extern "C" {
#include "hipfft_stub.h.inc"
}

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_STUB_H_
