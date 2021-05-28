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
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/cufft_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {
using ::testing::FloatNear;

TEST_P(Test, Dummy) {}  // Make INSTANTIATE_TEST_SUITE_P happy.

TEST_F(Test, ComplexToComplexTransform_1DCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());

  constexpr size_t kWindowSize = 256;
  constexpr size_t kWindowSizeBytes = kWindowSize * sizeof(cufftComplex);
  TFRT_ASSERT_AND_ASSIGN(
      auto host_data,
      CuMemHostAlloc(current, kWindowSizeBytes, CU_MEMHOSTALLOC_DEFAULT));
  for (size_t i = 0; i < kWindowSize; ++i) {
    static_cast<cufftComplex*>(host_data.get().raw())[i] = {
        static_cast<float>(2 * i), 0};
  }

  // Allocate enough memory for output to be written in place.
  TFRT_ASSERT_AND_ASSIGN(auto device_data,
                         CuMemAlloc(current, kWindowSizeBytes));

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         CuStreamCreate(current, CU_STREAM_DEFAULT));

  // Prepare FFT plan.
  TFRT_ASSERT_AND_ASSIGN(OwningCufftHandle plan,
                         CufftPlan1d(kWindowSize, CUFFT_C2C, /*batch=*/1));

  EXPECT_THAT(CufftSetStream(plan.get(), stream.get()), IsSuccess());

  // Copy data and do transform.
  EXPECT_THAT(CuMemcpyAsync(current, device_data.get(), host_data.get(),
                            kWindowSizeBytes, stream.get()),
              IsSuccess());
  EXPECT_THAT(CufftExecC2C(plan.get(),
                           static_cast<cufftComplex*>(device_data.get().raw()),
                           static_cast<cufftComplex*>(device_data.get().raw()),
                           FftDirection::kForward),
              IsSuccess());
  EXPECT_THAT(CufftExecC2C(plan.get(),
                           static_cast<cufftComplex*>(device_data.get().raw()),
                           static_cast<cufftComplex*>(device_data.get().raw()),
                           FftDirection::kInverse),
              IsSuccess());
  EXPECT_THAT(CuMemcpyAsync(current, host_data.get(), device_data.get(),
                            kWindowSizeBytes, stream.get()),
              IsSuccess());

  EXPECT_THAT(CuStreamSynchronize(stream.get()), IsSuccess());

  for (size_t i = 0; i < kWindowSize; ++i) {
    const float2 element = static_cast<cufftComplex*>(host_data.get().raw())[i];
    EXPECT_THAT(element.x, FloatNear(kWindowSize * 2 * i, 0.1));
    EXPECT_THAT(element.y, FloatNear(0, 0.1));
  }
  EXPECT_THAT(CufftDestroy(plan.get()), IsSuccess());
}

TEST_F(Test, RealToComplexTransform_1D_PlanManyCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());

  constexpr size_t kWindowSize = 256;
  constexpr size_t kWindowSizeBytesInput = kWindowSize * sizeof(cufftReal);
  constexpr size_t kWindowSizeBytesOutput = kWindowSize * sizeof(cufftComplex);

  // Allocate enough for reuse as output.
  TFRT_ASSERT_AND_ASSIGN(
      auto host_data,
      CuMemHostAlloc(current, kWindowSizeBytesOutput, CU_MEMHOSTALLOC_DEFAULT));

  const float kPi = std::acos(-1);
  for (size_t i = 0; i < kWindowSize; ++i) {
    static_cast<cufftReal*>(host_data.get().raw())[i] =
        static_cast<float>(std::sin(2 * kPi * i / kWindowSize));
  }

  // Allocate enough memory for output to be written in place.
  TFRT_ASSERT_AND_ASSIGN(auto device_data,
                         CuMemAlloc(current, kWindowSizeBytesOutput));

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         CuStreamCreate(current, CU_STREAM_DEFAULT));

  // Prepare FFT plan.
  llvm::SmallVector<int, 3> dims = {kWindowSize};
  // llvm::SmallVector<int, 3> input_embed = {};
  CufftManyOptions<int> options;
  options.rank = 1;
  options.dims = dims;
  options.input_dist = 1;
  options.input_embed = {};
  options.input_stride = 0;
  options.output_dist = 1;
  options.output_embed = {};
  options.output_stride = 0;

  TFRT_ASSERT_AND_ASSIGN(OwningCufftHandle plan,
                         CufftPlanMany(CUFFT_R2C, /*batch=*/1, options));
  EXPECT_THAT(CufftSetStream(plan.get(), stream.get()), IsSuccess());

  // Copy data and do transform.
  EXPECT_THAT(CuMemcpyAsync(current, device_data.get(), host_data.get(),
                            kWindowSizeBytesInput, stream.get()),
              IsSuccess());
  EXPECT_THAT(
      CufftExecR2C(plan.get(), static_cast<cufftReal*>(device_data.get().raw()),
                   static_cast<cufftComplex*>(device_data.get().raw())),
      IsSuccess());
  EXPECT_THAT(CuMemcpyAsync(current, host_data.get(), device_data.get(),
                            kWindowSizeBytesOutput, stream.get()),
              IsSuccess());

  EXPECT_THAT(CuStreamSynchronize(stream.get()), IsSuccess());

  float2* elements = static_cast<cufftComplex*>(host_data.get().raw());
  EXPECT_THAT(elements[1].y,
              FloatNear(-1 * static_cast<float>(kWindowSize) / 2, 0.1));
  for (size_t i = 0; i < kWindowSize; ++i) {
    if (i == 1) continue;
    EXPECT_THAT(elements[i].x, FloatNear(0, 0.1));
    EXPECT_THAT(elements[i].y, FloatNear(0, 0.1));
  }
  EXPECT_THAT(CufftDestroy(plan.get()), IsSuccess());
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
