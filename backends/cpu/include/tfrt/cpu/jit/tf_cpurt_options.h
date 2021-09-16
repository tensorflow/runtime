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

#ifndef TFRT_BACKENDS_CPU_JIT_TF_CPURT_OPTIONS_H_
#define TFRT_BACKENDS_CPU_JIT_TF_CPURT_OPTIONS_H_

// Options for TF CPU RT pipeline that lowers TF dialect to Linalg on buffers.

namespace tfrt {
namespace cpu {
namespace jit {

struct TfCpuRtOptions {
  // Enable codegen strategy for reductions.
  bool codegen_reductions = false;

  // Enable codegen strategy for elemenwise ops.
  bool codegen_cwise = false;
};

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_TF_CPURT_OPTIONS_H_
