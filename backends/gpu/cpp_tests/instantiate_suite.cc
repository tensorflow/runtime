// Copyright 2020 The TensorFlow Runtime Authors
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

// Instantiates parameterized GPU wrapper tests for each platform.

#include "common.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

static std::string ToString(const ::testing::TestParamInfo<Platform>& info) {
  std::string buffer;
  llvm::raw_string_ostream(buffer) << info.param;
  return buffer;
}

// Note: ROCm platform is not fully supported yet.
static auto kParams = testing::Values(Platform::CUDA);

INSTANTIATE_TEST_SUITE_P(Suite, Test, kParams, ToString);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
