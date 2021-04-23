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

// Mostly auto-generated rocFFT API header.
#ifndef TFRT_GPU_WRAPPER_ROCFFT_STUB_H_
#define TFRT_GPU_WRAPPER_ROCFFT_STUB_H_

#include <cstddef>

using rocfft_plan = struct rocfft_plan_t*;
using rocfft_plan_description = struct rocfft_plan_description_t*;
using rocfft_execution_info = struct rocfft_execution_info_t*;

extern "C" {
#include "rocfft_stub.h.inc"
}

#endif  // TFRT_GPU_WRAPPER_ROCFFT_STUB_H_
