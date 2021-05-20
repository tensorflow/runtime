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
#include <llvm/Support/Errc.h>

#include <ostream>

#include "gtest/gtest.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {

class GpuBufferTest : public ::testing::TestWithParam<wrapper::Platform> {};

TEST_P(GpuBufferTest, Basic) {
  ASSERT_TRUE(IsSuccess(Init(GetParam())));
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  wrapper::OwningStream stream;
  {
    TFRT_ASSERT_AND_ASSIGN(auto current, wrapper::CtxSetCurrent(context.get()));
    TFRT_ASSERT_AND_ASSIGN(
        stream, wrapper::StreamCreate(current, wrapper::StreamFlags::DEFAULT));
  }
  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(std::move(context));
  auto gpu_allocator =
      MakeAvailableAsyncValueRef<GpuDefaultAllocator>(gpu_context.CopyRef());

  size_t buffer_size = 512;
  TFRT_ASSERT_AND_ASSIGN(
      auto gpu_buffer,
      GpuBuffer::Allocate(gpu_allocator.CopyRef(), buffer_size, stream.get()));

  stream.reset();  // Destroy `stream` before `gpu_context`.
}

INSTANTIATE_TEST_SUITE_P(BaseTestCases, GpuBufferTest,
                         ::testing::Values(wrapper::Platform::CUDA));

}  // namespace gpu
}  // namespace tfrt
