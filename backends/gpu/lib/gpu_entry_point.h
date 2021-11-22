// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Provides constants shared between the compiler and gpu executor.

#ifndef TFRT_GPU_ENTRY_POINT_H_
#define TFRT_GPU_ENTRY_POINT_H_

#include <cstdint>

namespace tfrt {
namespace gpu {

constexpr const char* GetEntryPointFuncName() {
  return "get_tfrt_gpu_entry_point";
}

constexpr const char* GetEntryPointOpName() {
  return "tfrt_gpu.get_entry_point";
}

constexpr int64_t GetEntryPointVersion() {
  // CL number of the last ABI change. This needs to be bumped manually.
  // TODO(b/206463680): implement proper/automatic versioning.
  return 410724080;
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_ENTRY_POINT_H_
