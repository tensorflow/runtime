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

// Unit test for runtime API (abstraction layer for CUDA and HIP runtime).

#include "tfrt/gpu/wrapper/runtime_wrapper.h"

#include "common.h"
#include "tfrt/gpu/wrapper/cudart_wrapper.h"
#include "tfrt/gpu/wrapper/hip_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

TEST_P(Test, RuntimeInit) {
  auto platform = GetParam();
  EXPECT_THAT(Init(platform), IsSuccess());
  EXPECT_THAT(CtxSetCurrent({nullptr, platform}).takeError(), IsSuccess());
  EXPECT_THAT(Free(nullptr, platform), IsSuccess());
}

TEST_P(Test, RuntimeVersion) {
  auto platform = GetParam();
  TFRT_ASSERT_AND_ASSIGN(auto version, RuntimeGetVersion(platform));
  EXPECT_GT(version, 0);
}

TEST_F(Test, DevicePropertiesCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto dev_props, CudaGetDeviceProperties(current));
  ASSERT_GT(dev_props.major, 0);
}

TEST_F(Test, DevicePropertiesROCm) {
  auto platform = Platform::ROCm;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto dev_props, HipGetDeviceProperties(current));
  ASSERT_GT(dev_props.major, 0);
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
