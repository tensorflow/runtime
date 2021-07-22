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

#ifndef TFRT_GPU_WRAPPER_CCL_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CCL_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "src/nccl.h"  // from @nccl_headers
#include "tfrt/gpu/wrapper/cuda_forwards.h"
#include "tfrt/gpu/wrapper/wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template <>
Expected<ncclDataType_t> Parse<ncclDataType_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclDataType_t value);

template <>
Expected<ncclRedOp_t> Parse<ncclRedOp_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclRedOp_t value);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CCL_WRAPPER_H_
