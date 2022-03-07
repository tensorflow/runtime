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

// Mostly auto-generated rocBLAS API header.
#ifndef TFRT_GPU_WRAPPER_ROCBLAS_STUB_H_
#define TFRT_GPU_WRAPPER_ROCBLAS_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"

#ifdef _WIN32
#define ROCBLAS_EXPORT __declspec(dllexport)
#else
#define ROCBLAS_EXPORT __attribute__((visibility("default")))
#endif

// Declare types from rocblas.h used in rocblas_stub.h.inc.
using rocblas_int = int;
using rocblas_stride = int64_t;

struct rocblas_half {
  uint16_t data;
};

struct rocblas_float_complex {
  float x, y;
};

struct rocblas_double_complex {
  double x, y;
};

extern "C" {
#include "rocblas_stub.h.inc"
}

#endif  // TFRT_GPU_WRAPPER_ROCBLAS_STUB_H_
