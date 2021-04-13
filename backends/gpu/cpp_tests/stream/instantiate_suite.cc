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
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

INSTANTIATE_TEST_SUITE_P(Suite, Test,
                         testing::Values(Platform::CUDA /*, Platform::ROCm*/),
                         [](const auto& info) {
                           std::string buffer;
                           llvm::raw_string_ostream oss(buffer);
                           oss << info.param;
                           return oss.str();
                         });

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
