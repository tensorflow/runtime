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

// Unit test for SubAllocator.
#include "tfrt/gpu/memory/sub_allocator.h"

#include <llvm/Support/Errc.h>

#include <ostream>

#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/memory/gpu_allocator.h"

namespace tfrt {
namespace gpu {

// Google Test outputs to std::ostream. Provide ADL'able overloads.
template <typename T>
std::ostream& operator<<(std::ostream& os, T item) {
  llvm::raw_os_ostream raw_os(os);
  raw_os << item;
  return os;
}

class SubAllocatorTest
    : public ::testing::TestWithParam<gpu::stream::Platform> {};

TEST_P(SubAllocatorTest, GetPlatformTest) {
  EXPECT_EQ(SubAllocator(GetParam()).platform(), GetParam());
}

TEST_P(SubAllocatorTest, SingleStreamTest) {
  size_t kBlockSize = 512;
  size_t kAlignment = GpuAllocator::kAlignment;
  SubAllocator sub_allocator(GetParam());
  {
    TFRT_ASSERT_AND_ASSIGN(
        auto pointer,
        sub_allocator.Allocate(kBlockSize, gpu::stream::Stream()));
    uintptr_t address =
        reinterpret_cast<uintptr_t>(pointer.raw(gpu::stream::Platform::CUDA));
    ASSERT_EQ(address, kAlignment);
  }
  {
    TFRT_ASSERT_AND_ASSIGN(
        auto pointer,
        sub_allocator.Allocate(kBlockSize, gpu::stream::Stream()));
    uintptr_t address =
        reinterpret_cast<uintptr_t>(pointer.raw(gpu::stream::Platform::CUDA));
    ASSERT_EQ(address, 512 + kAlignment);
  }
}

INSTANTIATE_TEST_SUITE_P(BaseTestCases, SubAllocatorTest,
                         ::testing::Values(gpu::stream::Platform::CUDA));

}  // namespace gpu
}  // namespace tfrt
