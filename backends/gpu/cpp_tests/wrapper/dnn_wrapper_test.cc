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

#include "common.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

TEST_P(Test, DnnHandel) {
  auto platform = GetParam();
  EXPECT_TRUE(IsSuccess(Init(platform)));
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
  EXPECT_TRUE(IsSuccess(DnnSetTensorDescriptor(descriptor.get(),
                                               DnnDataType(0, platform),
                                               {2, 2, 3, 1}, {1, 2, 4, 12})));
}

TEST_F(Test, CudnnLogCUDA) {
  auto platform = Platform::CUDA;
  EXPECT_TRUE(IsSuccess(Init(platform)));
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
