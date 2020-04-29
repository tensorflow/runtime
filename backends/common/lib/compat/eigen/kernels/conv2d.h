/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===- conv2d.h -------------------------------------------------*- C++ -*-===//
//
// Conv2D kernel implementation using Eigen contraction.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_H_

#include <cstdint>

#include "../contraction_output_kernel.h"
#include "../spatial_convolution.h"
#include "conv2d_shape_functions.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {
namespace internal {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::BiasAddOutputKernel;
using ::tfrt::compat::SpatialConvolution;

inline llvm::Error CheckBias(const Conv2DParams& params,
                             const FixedRankShape<1>& bias_shape) {
  return CheckDimensionMatch("bias shape", bias_shape[0],
                             "output channels size", params.output_shape[3]);
}

inline llvm::Error CheckBatchNormArgs(const Conv2DParams& params,
                                      const FixedRankShape<1>& scale_shape,
                                      const FixedRankShape<1>& offset_shape,
                                      const FixedRankShape<1>& mean_shape,
                                      const FixedRankShape<1>& var_shape) {
  // All batch normalization arguments must match channels dimension.
  auto check = [&](string_view name, const FixedRankShape<1> shape) {
    return CheckDimensionMatch(name, shape[0], "output channels size",
                               params.output_shape[3]);
  };

  if (auto err = check("scale shape", scale_shape)) return err;
  if (auto err = check("offset shape", offset_shape)) return err;
  if (auto err = check("estimated mean shape", mean_shape)) return err;
  if (auto err = check("estimated var shape", var_shape)) return err;

  return llvm::Error::success();
}

template <typename T, typename OutputKernelBuilder>
inline void Conv2DImpl(const DHTIndexableView<T, 4>& input,
                       const DHTIndexableView<T, 4>& kernel,
                       const MutableDHTIndexableView<T, 4>& output,
                       Result<Chain> chain_out, StringAttribute padding,
                       ArrayAttribute<ssize_t> strides,
                       OutputKernelBuilder output_kernel_builder,
                       KernelErrorHandler handler, HostContext* host,
                       KernelFrame* frame) {
  // Validate convolution parameters.
  auto params = ComputeConv2DParams(input.FixedShape(), kernel.FixedShape(),
                                    padding.get(), {strides[0], strides[1]});

  TFRT_RETURN_IF_ERROR(handler, params.takeError());
  TFRT_RETURN_IF_ERROR(
      handler, CheckShapeMatch("output tensor shape", output.FixedShape(),
                               "computed output shape", params->output_shape));

  // Construct an output kernel from convolution parameters.
  auto output_kernel = output_kernel_builder(params.get());
  TFRT_RETURN_IF_ERROR(handler, output_kernel.takeError());

  auto on_done = [chain = chain_out.Allocate(),
                  frame = RAIIKernelFrame(*frame)]() { chain.emplace(); };

  const FixedRankShape<4>& kernel_shape = kernel.FixedShape();

  // 1x1 convolution can be computed as a simple Tensor contraction.
  if (kernel_shape[0] == 1 && kernel_shape[1] == 1 &&  // 1x1 kernel
      strides[0] == 1 && strides[1] == 1 &&            // 1x1 stride
      params->padding_type != PaddingType::kExplicit) {
    const ssize_t rest_size = params->output_shape[0] *  // batch
                              params->output_shape[1] *  // output height
                              params->output_shape[2];   // output width

    auto reshaped_in = FixedRankShape<2>({rest_size, kernel_shape[2]});
    auto reshaped_kern = FixedRankShape<2>({kernel_shape[2], kernel_shape[3]});
    auto reshaped_out = FixedRankShape<2>({rest_size, kernel_shape[3]});

    auto input_t = AsEigenConstTensor(input, reshaped_in);
    auto kernel_t = AsEigenConstTensor(kernel, reshaped_kern);
    auto output_t = AsEigenTensor(output, reshaped_out);

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dim({1, 0});
    auto expr = input_t.contract(kernel_t, contract_dim, output_kernel.get());

    AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                std::move(output_t), std::move(expr), std::move(on_done));

  } else {
    auto input_t = AsEigenConstTensor(input);
    auto filter_t = AsEigenConstTensor(kernel);
    auto output_t = AsEigenTensor(output);

    // clang-format off
    auto expr = SpatialConvolution(input_t, input.FixedShape(),
                                   filter_t, kernel.FixedShape(),
                                   /*strides=*/{strides[0], strides[1]},
                                   /*paddings=*/params->paddings,
                                   /*dilations=*/params->dilations,
                                   /*inflations=*/{1, 1},
                                   /*output_kernel=*/output_kernel.get());
    // clang-format on

    AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                std::move(output_t), std::move(expr), std::move(on_done));
  }
}

template <typename T>
static void Conv2D(ArgumentView<DHTIndexableView<T, 4>> input,
                   ArgumentView<DHTIndexableView<T, 4>> kernel,
                   ArgumentView<MutableDHTIndexableView<T, 4>> output,
                   Argument<Chain> chain_in, Result<Chain> chain_out,
                   StringAttribute padding, ArrayAttribute<ssize_t> strides,
                   KernelErrorHandler handler, HostContext* host,
                   KernelFrame* frame) {
  using OutputKernel = llvm::Expected<Eigen::NoOpOutputKernel>;

  auto output_kernel = [](Conv2DParams) -> OutputKernel {
    return Eigen::NoOpOutputKernel();
  };

  Conv2DImpl<T>(input.get(), kernel.get(), output.get(), chain_out, padding,
                strides, std::move(output_kernel), handler, host, frame);
}

template <typename T, typename Activation = Identity>
void Conv2DBatchNorm(ArgumentView<DHTIndexableView<T, 4>> input,
                     ArgumentView<DHTIndexableView<T, 4>> kernel,
                     ArgumentView<DHTIndexableView<T, 1>> scale,   // aka gamma
                     ArgumentView<DHTIndexableView<T, 1>> offset,  // aka beta
                     ArgumentView<DHTIndexableView<T, 1>> estimated_mean,
                     ArgumentView<DHTIndexableView<T, 1>> estimated_variance,
                     ArgumentView<MutableDHTIndexableView<T, 4>> output,
                     Argument<Chain> chain_in, Result<Chain> chain_out,
                     Attribute<float> epsilon, StringAttribute padding,
                     ArrayAttribute<ssize_t> strides,
                     KernelErrorHandler handler, HostContext* host,
                     KernelFrame* frame) {
  using OutputKernel = llvm::Expected<BatchNormOutputKernel<T, Activation>>;

  auto output_kernel = [&](Conv2DParams params) -> OutputKernel {
    if (auto err = CheckBatchNormArgs(
            params, scale->FixedShape(), offset->FixedShape(),
            estimated_mean->FixedShape(), estimated_variance->FixedShape())) {
      return std::move(err);
    }

    return BatchNormOutputKernel<T, Activation>(
        AsEigenConstTensor(scale.get()),               // gamma
        AsEigenConstTensor(offset.get()),              // beta
        AsEigenConstTensor(estimated_mean.get()),      // mean
        AsEigenConstTensor(estimated_variance.get()),  // variance
        epsilon.get());
  };

  Conv2DImpl<T>(input.get(), kernel.get(), output.get(), chain_out, padding,
                strides, std::move(output_kernel), handler, host, frame);
}

template <typename T, typename Activation = Identity>
void Conv2DBias(ArgumentView<DHTIndexableView<T, 4>> input,
                ArgumentView<DHTIndexableView<T, 4>> kernel,
                ArgumentView<DHTIndexableView<T, 1>> bias,
                ArgumentView<MutableDHTIndexableView<T, 4>> output,
                Argument<Chain> chain_in, Result<Chain> chain_out,
                StringAttribute padding, ArrayAttribute<ssize_t> strides,
                KernelErrorHandler handler, HostContext* host,
                KernelFrame* frame) {
  using OutputKernel = llvm::Expected<BiasAddOutputKernel<T, Activation>>;

  auto output_kernel = [bias](Conv2DParams params) -> OutputKernel {
    if (auto err = internal::CheckBias(params, bias->FixedShape())) {
      return std::move(err);
    }
    return BiasAddOutputKernel<T, Activation>(AsEigenConstTensor(bias.get()));
  };

  Conv2DImpl<T>(input.get(), kernel.get(), output.get(), chain_out, padding,
                strides, std::move(output_kernel), handler, host, frame);
}

}  // namespace internal
}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_H_
