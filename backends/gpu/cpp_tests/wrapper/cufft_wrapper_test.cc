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
#include "tfrt/gpu/wrapper/cufft_wrapper.h"

#include <cstddef>
#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"

#define TFRT_ASSERT_OK(expr) ASSERT_TRUE(IsSuccess(expr))

namespace tfrt {
namespace gpu {
namespace wrapper {
namespace {

TEST(CufftWrapperTest, ComplexToComplexTransform_1D) {
  TFRT_ASSERT_OK(Init(Platform::CUDA));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(Platform::CUDA));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(Platform::CUDA, 0));
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

  TFRT_ASSERT_OK(CufftSetStream(plan.get(), stream.get()));

  // Copy data and do transform.
  TFRT_ASSERT_OK(CuMemcpyAsync(current, device_data.get(), host_data.get(),
                               kWindowSizeBytes, stream.get()));
  TFRT_ASSERT_OK(CufftExecC2C(
      plan.get(), static_cast<cufftComplex*>(device_data.get().raw()),
      static_cast<cufftComplex*>(device_data.get().raw()),
      FftDirection::kForward));
  TFRT_ASSERT_OK(CufftExecC2C(
      plan.get(), static_cast<cufftComplex*>(device_data.get().raw()),
      static_cast<cufftComplex*>(device_data.get().raw()),
      FftDirection::kInverse));
  TFRT_ASSERT_OK(CuMemcpyAsync(current, host_data.get(), device_data.get(),
                               kWindowSizeBytes, stream.get()));

  TFRT_ASSERT_OK(CuStreamSynchronize(stream.get()));

  for (size_t i = 0; i < kWindowSize; ++i) {
    const float2 element = static_cast<cufftComplex*>(host_data.get().raw())[i];
    EXPECT_THAT(element.x, testing::FloatNear(kWindowSize * 2 * i, 0.1));
    EXPECT_THAT(element.y, testing::FloatNear(0, 0.1));
  }
  TFRT_ASSERT_OK(CufftDestroy(plan.get()));
}

TEST(CufftWrapperTest, RealToComplexTransform_1D_PlanMany) {
  TFRT_ASSERT_OK(Init(Platform::CUDA));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(Platform::CUDA));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(Platform::CUDA, 0));
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
  TFRT_ASSERT_OK(CufftSetStream(plan.get(), stream.get()));

  // Copy data and do transform.
  TFRT_ASSERT_OK(CuMemcpyAsync(current, device_data.get(), host_data.get(),
                               kWindowSizeBytesInput, stream.get()));
  TFRT_ASSERT_OK(
      CufftExecR2C(plan.get(), static_cast<cufftReal*>(device_data.get().raw()),
                   static_cast<cufftComplex*>(device_data.get().raw())));
  TFRT_ASSERT_OK(CuMemcpyAsync(current, host_data.get(), device_data.get(),
                               kWindowSizeBytesOutput, stream.get()));

  TFRT_ASSERT_OK(CuStreamSynchronize(stream.get()));

  float2* elements = static_cast<cufftComplex*>(host_data.get().raw());
  EXPECT_THAT(
      elements[1].y,
      testing::FloatNear(-1 * static_cast<float>(kWindowSize) / 2, 0.1));
  for (size_t i = 0; i < kWindowSize; ++i) {
    if (i == 1) continue;
    EXPECT_THAT(elements[i].x, testing::FloatNear(0, 0.1));
    EXPECT_THAT(elements[i].y, testing::FloatNear(0, 0.1));
  }
  TFRT_ASSERT_OK(CufftDestroy(plan.get()));
}

}  // namespace
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
