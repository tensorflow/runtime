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

// Unit test for cuFFT wrapper.
#include "tfrt/gpu/wrapper/fft_wrapper.h"

#include "common.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/cufft_wrapper.h"
#include "tfrt/gpu/wrapper/hipfft_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {
using ::testing::FloatNear;

TEST_P(Test, Dummy) {}  // Make INSTANTIATE_TEST_SUITE_P happy.

TEST_P(Test, RealToComplex1D) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto stream, StreamCreate(current, {}));

  constexpr size_t kWindowSize = 256;
  constexpr size_t kWindowSizeBytesInput = kWindowSize * sizeof(cufftReal);
  constexpr size_t kWindowSizeBytesOutput = kWindowSize * sizeof(cufftComplex);

  // Prepare FFT plan.

  llvm::SmallVector<int64_t, 3> dims = {kWindowSize};
  llvm::SmallVector<int64_t, 3> input_embed = {};
  llvm::SmallVector<int64_t, 3> output_embed = {};
  auto rank = 1;
  auto input_dist = 1;
  auto input_stride = 0;
  auto output_dist = 1;
  auto output_stride = 0;

  TFRT_ASSERT_AND_ASSIGN(OwningFftHandle plan, FftCreate(platform));
  EXPECT_THAT(FftMakePlanMany(plan.get(), FftType::kR2C, /*batch=*/1, rank,
                              dims, input_embed, input_stride, output_embed,
                              output_stride, input_dist, output_dist)
                  .takeError(),
              IsSuccess());
  EXPECT_THAT(FftSetStream(plan.get(), stream.get()), IsSuccess());

  // Allocate enough for reuse as output.
  TFRT_ASSERT_AND_ASSIGN(auto host_data,
                         MemHostAlloc(current, kWindowSizeBytesOutput, {}));

  const float kPi = std::acos(-1);
  for (size_t i = 0; i < kWindowSize; ++i) {
    static_cast<cufftReal*>(host_data.get().raw())[i] =
        static_cast<float>(std::sin(2 * kPi * i / kWindowSize));
  }

  // Allocate enough memory for output to be written in place.
  TFRT_ASSERT_AND_ASSIGN(auto device_data,
                         MemAlloc(current, kWindowSizeBytesOutput));

  // Copy data and do transform.
  EXPECT_THAT(MemcpyAsync(current, device_data.get(), host_data.get(),
                          kWindowSizeBytesInput, stream.get()),
              IsSuccess());
  EXPECT_THAT(
      FftExec(plan.get(), static_cast<Pointer<cufftReal>>(device_data.get()),
              static_cast<Pointer<cufftComplex>>(device_data.get()),
              FftType::kR2C),
      IsSuccess());
  EXPECT_THAT(MemcpyAsync(current, host_data.get(), device_data.get(),
                          kWindowSizeBytesOutput, stream.get()),
              IsSuccess());

  EXPECT_THAT(StreamSynchronize(stream.get()), IsSuccess());

  float2* elements = static_cast<cufftComplex*>(host_data.get().raw());
  EXPECT_THAT(elements[1].y,
              FloatNear(-1 * static_cast<float>(kWindowSize) / 2, 0.1));
  for (size_t i = 0; i < kWindowSize; ++i) {
    if (i == 1) continue;
    EXPECT_THAT(elements[i].x, FloatNear(0, 0.1));
    EXPECT_THAT(elements[i].y, FloatNear(0, 0.1));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
