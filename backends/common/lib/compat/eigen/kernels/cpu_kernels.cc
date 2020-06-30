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

//===- cpu_kernels.cc -----------------------------------------------------===//
//
// This file registers a few kernels implemented using the Eigen library.
//
//===----------------------------------------------------------------------===//

#include "batch_norm.h"
#include "conv2d.h"
#include "max_pooling.h"
#include "zero_padding.h"

namespace tfrt {
namespace compat {

template <typename T>
static AsyncValueRef<Chain> ZeroPadding(const DenseHostTensor& input,
                                        DenseHostTensor* output,
                                        const Chain& chain_in,
                                        ArrayAttribute<ssize_t> padding,
                                        const ExecutionContext& exec_ctx) {
  if (padding.size() != 2) {
    return EmitErrorAsync(exec_ctx, "ZeroPadding expects padding of length 2");
  }
  int32_t height = static_cast<int32_t>(padding[0]);
  int32_t width = static_cast<int32_t>(padding[1]);

  return TfPadImpl<T>(input, height, height, width, width, output, exec_ctx);
}

template <typename T>
static AsyncValueRef<Chain> MaxPool2D(const DenseHostTensor& input,
                                      DenseHostTensor* output, Chain chain_in,
                                      StringAttribute padding,
                                      ArrayAttribute<ssize_t> ksize,
                                      ArrayAttribute<ssize_t> strides,
                                      const ExecutionContext& exec_ctx) {
  return MaxPoolImpl<T>(input, output, padding.get(), strides.data(),
                        ksize.data(), exec_ctx);
}

template <typename T>
static AsyncValueRef<Chain> Conv2D(const DenseHostTensor& input,
                                   const DenseHostTensor& filter,
                                   DenseHostTensor* output, Chain chain_in,
                                   StringAttribute padding,
                                   ArrayAttribute<ssize_t> strides,
                                   const ExecutionContext& exec_ctx) {
  using OutputKernel = llvm::Expected<Eigen::NoOpOutputKernel>;
  auto output_kernel = [](Conv2DParams) -> OutputKernel {
    return Eigen::NoOpOutputKernel();
  };

  return internal::Conv2DImpl<T>(input, filter, output, padding.get(),
                                 strides.data(), std::move(output_kernel),
                                 exec_ctx);
}

template <typename T>
static AsyncValueRef<Chain> FusedBatchNormV3Kernel(
    DenseHostTensor* input, const DenseHostTensor& scale,
    const DenseHostTensor& bias, const DenseHostTensor& mean,
    const DenseHostTensor& variance, Chain chain_in, Attribute<float> epsilon,
    const ExecutionContext& exec_ctx) {
  return FusedBatchNormV3Impl<T>(*input, scale, bias, mean, variance, input,
                                 epsilon.get(), exec_ctx);
}

}  // namespace compat

void RegisterEigenKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.zero_padding.f32",
                      TFRT_KERNEL(compat::ZeroPadding<float>));
  registry->AddKernel("eigen.max_pooling_2d.f32",
                      TFRT_KERNEL(compat::MaxPool2D<float>));
  registry->AddKernel("eigen.conv2d.f32", TFRT_KERNEL(compat::Conv2D<float>));
  registry->AddKernel("eigen.batch_norm.f32",
                      TFRT_KERNEL(compat::FusedBatchNormV3Kernel<float>));
  registry->AddKernel("eigen.conv2d.batch_norm.f32",
                      TFRT_KERNEL(compat::internal::Conv2DBatchNorm<float>));
  registry->AddKernel(
      "eigen.conv2d.batch_norm.relu.f32",
      TFRT_KERNEL(compat::internal::Conv2DBatchNorm<float, compat::Relu>));
  registry->AddKernel("eigen.conv2d.bias.f32",
                      TFRT_KERNEL(compat::internal::Conv2DBias<float>));
}

}  // namespace tfrt
