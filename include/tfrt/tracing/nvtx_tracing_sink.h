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

#ifndef TFRT_TRACING_NVTX_TRACING_SINK_H_
#define TFRT_TRACING_NVTX_TRACING_SINK_H_

#include "nvtx3/nvToolsExt.h"  // from @cuda_headers

namespace tfrt {
namespace tracing {

// Sets the function to inject the backend into NVTX. A CUPTI-based tracing
// backend can call this function (with InitializeInjectionNvtx2 defined by
// CUPTI) so that NVTX activities are routed correctly.
//
// Alternatively, one can define the NVTX_INJECTION64_PATH env variable to point
// to a shared library implementing the InitializeInjectionNvtx2 function. This
// is how Nsight injects CUPTI.
//
// Injection needs to happen before tracing is enabled. Please be aware of the
// static initialization order fiasco.
void SetNvtxInjectionFunc(NvtxInitializeInjectionNvtxFunc_t func);

}  // namespace tracing
}  // namespace tfrt

#endif  // TFRT_TRACING_NVTX_TRACING_SINK_H_
