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

// Unit test for GPU types.

#include "tfrt/gpu/gpu_types.h"

#include "common.h"
#include "gmock/gmock.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;
using wrapper::Test;

TEST_P(Test, GpuContext) {
  ASSERT_THAT(Init(GetParam()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto device, wrapper::DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(
      auto context, wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device));

  GpuContext gpu_context(std::move(context));
  gpu_context.release();
  EXPECT_THAT(gpu_context.get(), IsNull());
}

TEST_P(Test, GpuStream) {
  ASSERT_THAT(Init(GetParam()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto device, wrapper::DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(
      auto context, wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, wrapper::CtxGetCurrent());

  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(std::move(context));

  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      wrapper::StreamCreate(current, wrapper::StreamFlags::DEFAULT));

  GpuStream gpu_stream(gpu_context.CopyRef(), std::move(stream));
  EXPECT_THAT(gpu_stream.get(), NotNull());
  gpu_stream.release();
  EXPECT_THAT(gpu_stream.get(), IsNull());
}

TEST_P(Test, BorrowedGpuStream) {
  ASSERT_THAT(Init(GetParam()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto device, wrapper::DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(
      auto context, wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, wrapper::CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      wrapper::StreamCreate(current, wrapper::StreamFlags::DEFAULT));

  BorrowedGpuStream borrowed_stream(context.get(), stream.get());
  auto stream_ref = static_cast<AsyncValueRef<GpuStream>>(borrowed_stream);
  EXPECT_EQ(stream_ref->context()->get(), context.get());
  EXPECT_EQ(stream_ref->get(), stream.get());
}

TEST_P(Test, GpuEvent) {
  ASSERT_THAT(Init(GetParam()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto device, wrapper::DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(
      auto context, wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, wrapper::CtxGetCurrent());

  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(std::move(context));

  TFRT_ASSERT_AND_ASSIGN(
      auto event, wrapper::EventCreate(current, wrapper::EventFlags::DEFAULT));

  GpuEvent gpu_event(gpu_context.CopyRef(), std::move(event));
  EXPECT_THAT(gpu_event.get(), NotNull());
}

TEST_P(Test, GpuBuffer) {
  ASSERT_THAT(Init(GetParam()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto device, wrapper::DeviceGet(GetParam(), 0));
  TFRT_ASSERT_AND_ASSIGN(
      auto context, wrapper::CtxCreate(wrapper::CtxFlags::SCHED_AUTO, device));

  auto gpu_context = MakeAvailableAsyncValueRef<GpuContext>(std::move(context));
  auto gpu_allocator =
      MakeAvailableAsyncValueRef<GpuDefaultAllocator>(gpu_context.CopyRef());

  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      wrapper::StreamCreate(wrapper::Current(), wrapper::StreamFlags::DEFAULT));

  size_t buffer_size = 512;
  TFRT_ASSERT_AND_ASSIGN(
      auto gpu_buffer,
      GpuBuffer::Allocate(gpu_allocator.CopyRef(), buffer_size, stream.get()));
}

TEST_P(Test, GpuBufferWithGpuOneShotAllocator) {
  GpuPointer pointer(reinterpret_cast<void*>(0x0bef0000), GetParam());
  auto gpu_allocator =
      MakeAvailableAsyncValueRef<GpuOneShotAllocator<void>>(pointer);

  // 1. Allocation with zero sized buffer succeeds.
  TFRT_ASSERT_AND_ASSIGN(
      auto buffer1, GpuBuffer::Allocate(gpu_allocator.CopyRef(), /*size=*/0));
  EXPECT_EQ(buffer1.pointer(), GpuPointer(nullptr, GetParam()));

  // 2. Allocation with non-zero sized buffer succeeds - the previous allocation
  // does not nullify the pointer.
  TFRT_ASSERT_AND_ASSIGN(
      auto buffer2, GpuBuffer::Allocate(gpu_allocator.CopyRef(), /*size=*/1));
  EXPECT_EQ(buffer2.pointer(), pointer);

  // 3. Allocation with zero sized buffer succeeds while the pointer is null.
  TFRT_ASSERT_AND_ASSIGN(
      auto buffer3, GpuBuffer::Allocate(gpu_allocator.CopyRef(), /*size=*/0));
  EXPECT_EQ(buffer3.pointer(), GpuPointer(nullptr, GetParam()));

  // 4. Allocation with non-zero sized buffer fails while the pointer is null.
  std::string error_str = StrCat(
      GpuBuffer::Allocate(gpu_allocator.CopyRef(), /*size=*/1).takeError());
  EXPECT_THAT(
      error_str,
      HasSubstr(
          "Trying to allocate from GpuOneShotAllocator with null pointer"));

  // 5. Deallocations succeed (a non-zero sized buffer followed by a zero sized
  // buffer).
  ASSERT_THAT(buffer2.Deallocate(), IsSuccess());
  ASSERT_THAT(buffer3.Deallocate(), IsSuccess());

  // 6. Allocation with non-zero sized buffer succeeds after deallocations.
  // This also verifies that the previous deallocation of zero sized buffer does
  // not nullify the pointer.
  EXPECT_THAT(
      GpuBuffer::Allocate(gpu_allocator.CopyRef(), /*size=*/1).takeError(),
      IsSuccess());
}

}  // namespace gpu
}  // namespace tfrt
