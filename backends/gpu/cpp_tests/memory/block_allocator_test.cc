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

// Unit test for block allocator.
#include "tfrt/gpu/memory/block_allocator.h"

#include <llvm/Support/Errc.h>

#include <ostream>

#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {

class BlockAllocatorTest : public ::testing::TestWithParam<SubAllocator> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(IsSuccess(Init(wrapper::Platform::CUDA)));
  }
  llvm::Error ValidateBuffer(const gpu::GpuCrtBuffer* buffer,
                             size_t expected_size, uintptr_t expected_address) {
    if (!buffer->IsValid()) {
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "Buffer is not valid.");
    }
    if (buffer->size() != expected_size) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          llvm::formatv("Expected buffer size {0} but got {1}.", expected_size,
                        buffer->size()));
    }
    const auto address = reinterpret_cast<uintptr_t>(
        buffer->pointer().raw(wrapper::Platform::CUDA));
    if (address != expected_address) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          llvm::formatv("Expected buffer address {0} but got {1}.",
                        expected_address, address));
    }
    return llvm::Error::success();
  }

  BlockAllocator CreateSimpleBlockAllocator() {
    return BlockAllocator(&sub_allocator_);
  }

  SubAllocator sub_allocator_ = GetParam();
};

TEST_P(BlockAllocatorTest, SingleStreamTest) {
  BlockAllocator block_allocator = CreateSimpleBlockAllocator();
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(wrapper::Platform::CUDA, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current_context,
                         wrapper::CtxSetCurrent(context.get()));
  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      wrapper::StreamCreate(current_context, wrapper::StreamFlags::DEFAULT));
  // Validate that two buffers pointing at different blocks
  // can exists at the same time.
  {
    size_t large_buffer_size = 512;
    TFRT_ASSERT_AND_ASSIGN(
        auto large_buffer,
        block_allocator.Allocate(large_buffer_size, stream.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(large_buffer.get(), large_buffer_size,
                       /*expected_address=*/GpuCrtAllocator::kAlignment)));

    size_t small_buffer_size = 64;
    TFRT_ASSERT_AND_ASSIGN(
        auto small_buffer,
        block_allocator.Allocate(small_buffer_size, stream.get()));
    ASSERT_TRUE(IsSuccess(ValidateBuffer(
        small_buffer.get(), small_buffer_size,
        /*expected_address=*/512 + GpuCrtAllocator::kAlignment)));
  }
  // Allocate another buffer that should fit into already existing block.
  {
    size_t buffer_size = 256;
    TFRT_ASSERT_AND_ASSIGN(auto buffer,
                           block_allocator.Allocate(buffer_size, stream.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(buffer.get(), /*expected_size=*/512,
                       /*expected_address=*/GpuCrtAllocator::kAlignment)));
  }
  // Allocate another buffer that should fit into already existing block.
  {
    size_t buffer_size = 16;
    TFRT_ASSERT_AND_ASSIGN(auto buffer,
                           block_allocator.Allocate(buffer_size, stream.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(buffer.get(), /*expected_size=*/64,
                       /*expected_address=*/3 * GpuCrtAllocator::kAlignment)));
  }
  // Allocate another buffer that should result in a creation of another block.
  {
    size_t buffer_size = 1024;
    TFRT_ASSERT_AND_ASSIGN(auto buffer,
                           block_allocator.Allocate(buffer_size, stream.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(buffer.get(), buffer_size,
                       /*expected_address=*/4 * GpuCrtAllocator::kAlignment)));
  }
}

TEST_P(BlockAllocatorTest, MultipleStreamsTest) {
  BlockAllocator block_allocator = CreateSimpleBlockAllocator();
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(wrapper::Platform::CUDA, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current_context,
                         wrapper::CtxSetCurrent(context.get()));

  TFRT_ASSERT_AND_ASSIGN(
      auto stream_one,
      wrapper::StreamCreate(current_context, wrapper::StreamFlags::DEFAULT));
  TFRT_ASSERT_AND_ASSIGN(
      auto stream_two,
      wrapper::StreamCreate(current_context, wrapper::StreamFlags::DEFAULT));
  {
    size_t buffer_size = 256;
    TFRT_ASSERT_AND_ASSIGN(
        auto buffer, block_allocator.Allocate(buffer_size, stream_one.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(buffer.get(), buffer_size,
                       /*expected_address=*/GpuCrtAllocator::kAlignment)));
  }
  // Allocate another buffer in a different stream that should
  // result in a creation of a new block.
  {
    size_t buffer_size = 64;
    TFRT_ASSERT_AND_ASSIGN(
        auto buffer, block_allocator.Allocate(buffer_size, stream_two.get()));
    ASSERT_TRUE(IsSuccess(
        ValidateBuffer(buffer.get(), buffer_size,
                       /*expected_address=*/2 * GpuCrtAllocator::kAlignment)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BaseTestCases, BlockAllocatorTest,
    ::testing::Values(SubAllocator(wrapper::Platform::CUDA)));

}  // namespace gpu
}  // namespace tfrt
