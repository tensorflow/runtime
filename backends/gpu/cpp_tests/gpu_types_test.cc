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
using testing::IsNull;
using testing::NotNull;
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
  EXPECT_EQ(stream_ref->context(), context.get());
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

}  // namespace gpu
}  // namespace tfrt
