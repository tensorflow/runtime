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

#include "conv2d_shape_functions.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/contraction_output_kernel.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/kernels/shape_functions.h"
#include "tfrt/common/compat/eigen/spatial_convolution.h"
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
inline AsyncValueRef<Chain> Conv2DImpl(
    const DenseHostTensor& input, const DenseHostTensor& filter,
    DenseHostTensor* output, string_view padding, ArrayRef<ssize_t> strides,
    OutputKernelBuilder output_kernel_builder,
    const ExecutionContext& exec_ctx) {
  DHTIndexableView<T, 4> input_view(&input);
  DHTIndexableView<T, 4> filter_view(&filter);
  MutableDHTIndexableView<T, 4> output_view(output);

  if (strides.size() != 2) {
    return EmitErrorAsync(exec_ctx, "strides should have 2 elements");
  }

  // Validate convolution parameters.
  auto params =
      ComputeConv2DParams(input_view.FixedShape(), filter_view.FixedShape(),
                          padding, {strides[0], strides[1]});
  if (auto error = params.takeError()) {
    return EmitErrorAsync(exec_ctx, StrCat(error));
  }
  if (auto error =
          CheckShapeMatch("output tensor shape", output_view.FixedShape(),
                          "computed output shape", params->output_shape)) {
    return EmitErrorAsync(exec_ctx, StrCat(error));
  }

  // Construct an output kernel from convolution parameters.
  auto output_kernel = output_kernel_builder(params.get());
  if (auto error = output_kernel.takeError()) {
    return EmitErrorAsync(exec_ctx, StrCat(error));
  }

  const FixedRankShape<4>& kernel_shape = filter_view.FixedShape();

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

    auto input_t = AsEigenConstTensor(input_view, reshaped_in);
    auto kernel_t = AsEigenConstTensor(filter_view, reshaped_kern);
    auto output_t = AsEigenTensor(output_view, reshaped_out);

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dim({1, 0});
    auto expr = input_t.contract(kernel_t, contract_dim, output_kernel.get());

    return AsyncAssign(
        exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
        std::move(output_t), std::move(expr),
        KeepBuffers::alive(&input, &filter, output));
  } else {
    auto input_t = AsEigenConstTensor(input_view);
    auto filter_t = AsEigenConstTensor(filter_view);
    auto output_t = AsEigenTensor(output_view);

    // clang-format off
    auto expr = SpatialConvolution(input_t, input_view.FixedShape(),
                                   filter_t, filter_view.FixedShape(),
                                   /*strides=*/{strides[0], strides[1]},
                                   /*paddings=*/params->paddings,
                                   /*dilations=*/params->dilations,
                                   /*inflations=*/{1, 1},
                                   /*output_kernel=*/output_kernel.get());
    // clang-format on
    return AsyncAssign(
        exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
        std::move(output_t), std::move(expr),
        KeepBuffers::alive(&input, &filter, output));
  }
}

template <typename T, typename Activation = Identity>
AsyncValueRef<Chain> Conv2DBatchNorm(
    const DenseHostTensor& input, const DenseHostTensor& filter,
    const DenseHostTensor& scale,   // aka gamma
    const DenseHostTensor& offset,  // aka beta
    const DenseHostTensor& mean, const DenseHostTensor& variance,
    DenseHostTensor* output, Chain chain_in, Attribute<float> epsilon,
    StringAttribute padding, ArrayAttribute<ssize_t> strides,
    const ExecutionContext& exec_ctx) {
  using OutputKernel = llvm::Expected<BatchNormOutputKernel<T, Activation>>;

  auto output_kernel =
      [scale = scale.CopyRef(), offset = offset.CopyRef(),
       mean = mean.CopyRef(), variance = variance.CopyRef(),
       epsilon = epsilon.get()](Conv2DParams params) -> OutputKernel {
    DHTIndexableView<T, 1> scale_view(&scale);
    DHTIndexableView<T, 1> offset_view(&offset);
    DHTIndexableView<T, 1> mean_view(&mean);
    DHTIndexableView<T, 1> variance_view(&variance);
    if (auto err = CheckBatchNormArgs(
            params, scale_view.FixedShape(), offset_view.FixedShape(),
            mean_view.FixedShape(), variance_view.FixedShape())) {
      return std::move(err);
    }

    return BatchNormOutputKernel<T, Activation>(
        AsEigenConstTensor(scale_view),     // gamma
        AsEigenConstTensor(offset_view),    // beta
        AsEigenConstTensor(mean_view),      // mean
        AsEigenConstTensor(variance_view),  // variance
        epsilon);
  };

  return Conv2DImpl<T>(input, filter, output, padding.get(), strides.data(),
                       std::move(output_kernel), exec_ctx);
}

template <typename T, typename Activation = Identity>
AsyncValueRef<Chain> Conv2DBias(const DenseHostTensor& input,
                                const DenseHostTensor& filter,
                                const DenseHostTensor& bias,
                                DenseHostTensor* output, Chain chain_in,
                                StringAttribute padding,
                                ArrayAttribute<ssize_t> strides,
                                const ExecutionContext& exec_ctx) {
  using OutputKernel = llvm::Expected<BiasAddOutputKernel<T, Activation>>;

  auto output_kernel =
      [bias = bias.CopyRef()](Conv2DParams params) -> OutputKernel {
    DHTIndexableView<T, 1> bias_view(&bias);
    if (auto err = internal::CheckBias(params, bias_view.FixedShape())) {
      return std::move(err);
    }
    return BiasAddOutputKernel<T, Activation>(AsEigenConstTensor(bias_view));
  };

  return Conv2DImpl<T>(input, filter, output, padding.get(), strides.data(),
                       std::move(output_kernel), exec_ctx);
}

}  // namespace internal
}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_KERNELS_CONV2D_H_
