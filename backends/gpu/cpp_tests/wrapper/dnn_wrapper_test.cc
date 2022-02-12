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

#include "tfrt/gpu/wrapper/dnn_wrapper.h"

#include <cstdint>
#include <utility>

#include "common.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

TEST_P(Test, DnnHandel) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto handle, DnnCreate(current));
}

TEST_P(Test, DnnConvDesc) {
  auto platform = GetParam();
  TFRT_ASSERT_AND_ASSIGN(auto descriptor,
                         DnnCreateConvolutionDescriptor(platform));
}

TEST_P(Test, DnnTensorDesc) {
  auto platform = GetParam();
  TFRT_ASSERT_AND_ASSIGN(auto descriptor, DnnCreateTensorDescriptor(platform));
  EXPECT_THAT(DnnSetTensorDescriptor(descriptor.get(), DnnDataType(0, platform),
                                     {2, 2, 3, 1}, {1, 2, 4, 12}),
              IsSuccess());
}

TEST_F(Test, DnnConvFwdFloatCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto handle, DnnCreate(current));

  const int n = 4;   // batch count
  const int c = 8;   // input/output channels
  const int h = 16;  // input/output height
  const int w = 16;  // input/output width
  const int f = 3;   // filter width

  TFRT_ASSERT_AND_ASSIGN(auto tensor_desc, DnnCreateTensorDescriptor(platform));
  ASSERT_THAT(DnnSetTensorDescriptor(tensor_desc.get(), CUDNN_DATA_FLOAT,
                                     /*dims=*/{n, c, h, w},
                                     /*strides=*/{c * h * w, h * w, w, 1}),
              IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto filter_desc, DnnCreateFilterDescriptor(platform));
  ASSERT_THAT(DnnSetFilterDescriptor(filter_desc.get(), CUDNN_DATA_FLOAT,
                                     CUDNN_TENSOR_NCHW, /*dims=*/{c, c, f, f}),
              IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto conv_desc,
                         DnnCreateConvolutionDescriptor(platform));
  ASSERT_THAT(
      DnnSetConvolutionDescriptor(conv_desc.get(), /*pad=*/{1, 1},
                                  /*stride=*/{1, 1}, /*dilation=*/{1, 1},
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
      IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto input,
                         MemAlloc(current, n * c * h * w * sizeof(float)));
  TFRT_ASSERT_AND_ASSIGN(auto output,
                         MemAlloc(current, n * c * h * w * sizeof(float)));
  TFRT_ASSERT_AND_ASSIGN(auto filter,
                         MemAlloc(current, c * c * f * f * sizeof(float)));

  Pointer<void> workspace(nullptr, platform);
  ASSERT_THAT(DnnConvolutionForward(
                  current, handle.get(), CUDNN_DATA_FLOAT, tensor_desc.get(),
                  input.get(), filter_desc.get(), filter.get(), conv_desc.get(),
                  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace,
                  /*workspace_size_bytes=*/0, tensor_desc.get(), output.get()),
              IsSuccess());

  ASSERT_THAT(CtxSynchronize(current), IsSuccess());
}

TEST_F(Test, DnnConvFwdInt8x4CUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));

  TFRT_ASSERT_AND_ASSIGN(
      auto cc_major, CuDeviceGetAttribute(
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  TFRT_ASSERT_AND_ASSIGN(
      auto cc_minor, CuDeviceGetAttribute(
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  if (std::make_pair(cc_major, cc_minor) < std::make_pair(6, 1))
    GTEST_SKIP() << "No DP4A support on sm_" << cc_major << cc_minor;

  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto handle, DnnCreate(current));

  const int n = 4;   // batch count
  const int c = 8;   // input/output channels
  const int h = 16;  // input/output height
  const int w = 16;  // input/output width
  const int v = 4;   // vector width
  const int f = 3;   // filter width

  TFRT_ASSERT_AND_ASSIGN(auto tensor_desc, DnnCreateTensorDescriptor(platform));
  ASSERT_THAT(DnnSetTensorDescriptor(tensor_desc.get(), CUDNN_DATA_INT8x4,
                                     /*dims=*/{n, c / v, h, w},
                                     /*strides=*/{c / v * h * w, h * w, w, 1}),
              IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto filter_desc, DnnCreateFilterDescriptor(platform));
  ASSERT_THAT(
      DnnSetFilterDescriptor(filter_desc.get(), CUDNN_DATA_INT8x4,
                             CUDNN_TENSOR_NCHW_VECT_C, /*dims=*/{c, c, f, f}),
      IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto conv_desc,
                         DnnCreateConvolutionDescriptor(platform));
  ASSERT_THAT(
      DnnSetConvolutionDescriptor(conv_desc.get(), /*pad=*/{1, 1},
                                  /*stride=*/{1, 1}, /*dilation=*/{1, 1},
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32),
      IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto algos,
                         CudnnGetConvolutionForwardAlgorithm(
                             handle.get(), tensor_desc.get(), filter_desc.get(),
                             conv_desc.get(), tensor_desc.get(), 1));
  ASSERT_EQ(algos.front().status, CUDNN_STATUS_SUCCESS);

  ASSERT_THAT(
      CudnnSetConvolutionMathType(conv_desc.get(), algos.front().mathType),
      IsSuccess());
  auto algo = algos.front().algo;
  auto workspace_size_bytes = algos.front().memory;

  TFRT_ASSERT_AND_ASSIGN(auto input,
                         MemAlloc(current, n * c * h * w * sizeof(int8_t)));
  TFRT_ASSERT_AND_ASSIGN(auto output,
                         MemAlloc(current, n * c * h * w * sizeof(int8_t)));
  TFRT_ASSERT_AND_ASSIGN(auto filter,
                         MemAlloc(current, c * c * f * f * sizeof(int8_t)));

  DeviceMemory<void> workspace({nullptr, Platform::CUDA});
  if (workspace_size_bytes > 0) {
    TFRT_ASSERT_AND_ASSIGN(workspace, MemAlloc(current, workspace_size_bytes));
  }

  ASSERT_THAT(DnnConvolutionForward(
                  current, handle.get(), CUDNN_DATA_INT8x4, tensor_desc.get(),
                  input.get(), filter_desc.get(), filter.get(), conv_desc.get(),
                  algo, workspace.get(), workspace_size_bytes,
                  tensor_desc.get(), output.get()),
              IsSuccess());

  ASSERT_THAT(CtxSynchronize(current), IsSuccess());
}

TEST_F(Test, CudnnLogCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto handle, DnnCreate(current));
  TFRT_ASSERT_AND_ASSIGN(auto descriptor,
                         DnnCreateConvolutionDescriptor(platform));

  std::string log_string;
  llvm::raw_string_ostream(log_string)
      << DnnSetConvolutionGroupCount(descriptor.get(), -1);
  for (const char* substr : {"function cudnnSetConvolutionGroupCount() called",
                             "groupCount: type=int; val=-1"}) {
    EXPECT_TRUE(llvm::StringRef(log_string).contains(substr)) << log_string;
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
