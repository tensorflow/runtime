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

// Unit test for GpuBuffer.
#include "tfrt/gpu/memory/gpu_buffer.h"

#include <llvm/Support/Errc.h>

#include <ostream>

#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/memory/block_allocator.h"

namespace tfrt {
namespace gpu {

class GpuBufferTest : public ::testing::TestWithParam<SubAllocator> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(IsSuccess(Init(gpu::stream::Platform::CUDA)));
  }

  BlockAllocator CreateSimpleBlockAllocator() {
    return BlockAllocator(&sub_allocator_);
  }

  SubAllocator sub_allocator_ = GetParam();
};

TEST_P(GpuBufferTest, Basic) {
  BlockAllocator block_allocator = CreateSimpleBlockAllocator();
  TFRT_ASSERT_AND_ASSIGN(auto device,
                         DeviceGet(gpu::stream::Platform::CUDA, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current_context,
                         gpu::stream::CtxSetCurrent(context.get()));
  TFRT_ASSERT_AND_ASSIGN(
      auto stream, gpu::stream::StreamCreate(
                       current_context, gpu::stream::StreamFlags::DEFAULT));
  {
    // Allocation and deallocation via GpuAllocator.
    size_t buffer_size = 512;
    TFRT_ASSERT_AND_ASSIGN(auto buffer,
                           block_allocator.Allocate(buffer_size, stream.get()));
  }

  {
    size_t buffer_size = 512;
    TFRT_ASSERT_AND_ASSIGN(auto buffer,
                           block_allocator.Allocate(buffer_size, stream.get()));

    // Create GpuBuffer from externally allocated buffer, and deallocate via
    // deallocator.
    auto buffer_ptr = buffer.get();
    auto buffer2 =
        TakeRef(new GpuBuffer(buffer_ptr->pointer(), buffer_ptr->size(),
                              [buffer = std::move(buffer)](GpuBuffer*) {}));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BaseTestCases, GpuBufferTest,
    ::testing::Values(SubAllocator(gpu::stream::Platform::CUDA)));

}  // namespace gpu
}  // namespace tfrt
