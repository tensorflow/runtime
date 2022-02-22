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

// Mostly auto-generated MIOpen API header.
#ifndef TFRT_GPU_WRAPPER_MIOPEN_STUB_H_
#define TFRT_GPU_WRAPPER_MIOPEN_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"

#ifdef _WIN32
#define MIOPEN_EXPORT __declspec(dllexport)
#else
#define MIOPEN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
#include "miopen_stub.h.inc"
}

struct miopenConvAlgoPerf_t {
  union {
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;
  };

  float time;
  size_t memory;
};

struct miopenConvSolution_t {
  float time;
  size_t workspace_size;
  uint64_t solution_id;
  miopenConvAlgorithm_t algorithm;
};

const char* miopenGetErrorString(miopenStatus_t error);

#endif  // TFRT_GPU_WRAPPER_MIOPEN_STUB_H_
