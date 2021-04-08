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

// Unit test for DNN wrapper (abstraction layer for cuDNN and MIOpen).

#include "common.h"
#include "gtest/gtest.h"
#include "tfrt/gpu/stream/cudnn_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {
namespace stream {

TEST_P(Test, DnnHandel) {
  auto platform = GetParam();
  EXPECT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  EXPECT_TRUE(IsSuccess(DnnCreate(current).takeError()));
  TFRT_ASSERT_AND_ASSIGN(auto handle, DnnCreate(current));
  EXPECT_TRUE(IsSuccess(
      DnnCreateConvolutionDescriptor(current.platform()).takeError()));
}

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
