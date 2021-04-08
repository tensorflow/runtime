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

// CPU JIT implementation of TensorFlow ops
//
// This file contains helpers that register CPU specific JIT-compiled ops.

#ifndef TFRT_BACKENDS_CPU_OPS_TF_CPU_JIT_OPS_H_
#define TFRT_BACKENDS_CPU_OPS_TF_CPU_JIT_OPS_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {

class CpuOpRegistry;

void RegisterTfCpuJitOps(CpuOpRegistry* registry);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TF_CPU_JIT_OPS_H_
