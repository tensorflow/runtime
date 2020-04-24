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

//===- spatial_convolution.h ------------------------------------*- C++ -*-===//
//
// Spatial convolution implemented with Eigen expressions.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_H_

#include "contraction_kernel.h"
#include "spatial_convolution_data_mapper.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace compat {

template <typename Input, typename Kernel, typename OutputKernel>
struct SpatialConvolutionExpr {
  static constexpr int kRank = 4;

  using IndexType = typename Eigen::internal::traits<Input>::Index;

  // This is the Eigen expression template for the SpatialConvolution() declared
  // below. We do "template pattern matching" with template specializations to
  // provide custom performant data mappers, so we declare this type explicitly
  // to catch any changes to the generated AST at compile time.
  using type = Eigen::TensorReshapingOp<
      const Eigen::DSizes<IndexType, kRank>,
      const Eigen::TensorContractionOp<
          // Contraction dimensions.
          const Eigen::array<Eigen::IndexPair<IndexType>, 1>,
          // Reshape extracted image patches.
          const Eigen::TensorReshapingOp<
              const Eigen::DSizes<IndexType, 2>,
              const Eigen::TensorImagePatchOp<Eigen::Dynamic, Eigen::Dynamic,
                                              const Input>>,
          // Reshape kernel tensor.
          const Eigen::TensorReshapingOp<const Eigen::DSizes<IndexType, 2>,
                                         const Kernel>,
          const OutputKernel>>;
};

// SpatialConvolution() applies a 2D convolution to an input tensor expression
// `Input` using kernel expression `Kernel`.
//
// Input must be a 4 dimensional tensor in NHWC data format:
//   [batch_size, height, width, channels]
//
// Kernel must be a 4 dimensional tensor in HWIO data format:
//   [ filter_height, filter_width, in_channels, out_channels]
//
// It is also possible to add an elementwise output kernel to the contraction.
// The output kernel is called by Eigen when it "finalizes" the block of an
// output tensor.
template <typename Input, typename Kernel,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
typename SpatialConvolutionExpr<Input, Kernel, OutputKernel>::type
SpatialConvolution(const Input& input, const FixedRankShape<4>& input_shape,
                   const Kernel& kernel, const FixedRankShape<4>& kernel_shape,
                   const std::array<Eigen::Index, 2> strides,
                   const std::array<Eigen::Index, 4> paddings,
                   const std::array<Eigen::Index, 2> dilations = {1, 1},
                   const std::array<Eigen::Index, 2> inflations = {1, 1},
                   const OutputKernel& output_kernel = OutputKernel()) {
  // Spatial convolution requires 4 dimensional input and kernel.
  static constexpr int kRank =
      SpatialConvolutionExpr<Input, Kernel, OutputKernel>::kRank;

  using ::Eigen::Index;
  using ::Eigen::IndexPair;
  using ::Eigen::RowMajor;
  using ::Eigen::Tensor;
  using ::Eigen::TensorRef;
  using ::Eigen::internal::traits;

  static_assert(traits<Input>::NumDimensions == kRank,
                "Input must be 4 dimensional in NHWC data format");
  static_assert(traits<Kernel>::NumDimensions == kRank,
                "Kernel must be 4 dimensional in HWIO data format");

  static_assert(traits<Input>::Layout == RowMajor,
                "Input must be in RowMajor layout");
  static_assert(traits<Kernel>::Layout == RowMajor,
                "Kernel must be in RowMajor layout");

  using IndexType = typename traits<Input>::Index;
  static_assert(std::is_same<IndexType, typename traits<Kernel>::Index>::value,
                "Input and Kernel expressions must have the same Index type");

  using InputScalar = typename traits<Input>::Scalar;
  using KernelScalar = typename traits<Input>::Scalar;

  // Check input/kernel expressions dimensions. Constructing a TensorRef might
  // trigger expression evaluation, so we do it only in debug mode.
  using InputRef = TensorRef<Tensor<InputScalar, kRank, RowMajor, IndexType>>;
  using KernelRef = TensorRef<Tensor<KernelScalar, kRank, RowMajor, IndexType>>;

  auto ref_shape = [](auto ref) -> FixedRankShape<4> {
    return FixedRankShape<4>({ref.dimension(0), ref.dimension(1),
                              ref.dimension(2), ref.dimension(3)});
  };

  assert(ref_shape(InputRef(input)) == input_shape);
  assert(ref_shape(KernelRef(kernel)) == kernel_shape);
  (void)ref_shape;

  // Kernel is in HWIO data format.
  const IndexType kernel_filters = kernel_shape[3];   // output channels
  const IndexType kernel_channels = kernel_shape[2];  // input channels
  const IndexType kernel_width = kernel_shape[1];
  const IndexType kernel_height = kernel_shape[0];

  // Decode strides.
  const Eigen::Index height_stride = strides[0];
  const Eigen::Index width_stride = strides[1];

  // Decode paddings.
  const Eigen::Index padding_top = paddings[0];
  const Eigen::Index padding_bottom = paddings[1];
  const Eigen::Index padding_left = paddings[2];
  const Eigen::Index padding_right = paddings[3];

  // Decode dilations.
  const Eigen::Index height_dilation = dilations[0];
  const Eigen::Index width_dilation = dilations[1];

  // Decode inflations.
  const Eigen::Index heigh_inflation = inflations[0];
  const Eigen::Index width_inflation = inflations[1];

  // Input is in NHWC data format.
  const IndexType input_channels = input_shape[3];
  const IndexType input_width = input_shape[2];
  const IndexType input_height = input_shape[1];
  const IndexType input_batch = input_shape[0];

  // Kernel channels must match input channels.
  assert(input_channels == kernel_channels);
  (void)input_channels;

  // Effective kernel dimensions after applying dilations.
  const Index kernel_eff_width =
      kernel_width + (kernel_width - 1) * (width_dilation - 1);
  const Index kernel_eff_height =
      kernel_height + (kernel_height - 1) * (height_dilation - 1);

  // Effective input dimensions after applying paddings and inflations.
  // clang-format off
  const IndexType input_eff_height =
      input_height + (input_height - 1) * (heigh_inflation - 1) +
      padding_top + padding_bottom;
  const IndexType input_eff_width =
      input_width + (input_width - 1) * (width_inflation - 1) +
      padding_left + padding_right;
  // clang-format on

  // Compute output dimensions using effective input and kernel.
  const IndexType out_width =
      (input_eff_width - kernel_eff_width) / width_stride + 1;
  const IndexType out_height =
      (input_eff_height - kernel_eff_height) / height_stride + 1;

  // Molds the output of the patch extraction into a 2d tensor.
  Eigen::DSizes<IndexType, 2> pre_contract_dims(
      input_batch * out_height * out_width,
      kernel_height * kernel_width * kernel_channels);

  // Molds kernel into a 2d tensor.
  Eigen::DSizes<IndexType, 2> kernel_dims(
      kernel_height * kernel_width * kernel_channels, kernel_filters);

  // Output of the contraction (convolution).
  Eigen::DSizes<IndexType, kRank> post_contract_dims(input_batch, out_height,
                                                     out_width, kernel_filters);

  // Contract along the [kernel_h x kernel_w x kernel_channels] dimension.
  Eigen::array<IndexPair<IndexType>, 1> contract_dims(
      IndexPair<IndexType>(1, 0));

  // Patch row and column dimensions in Eigen are defined in the wrong order
  // (in NWHC data format). We pass `kernel_width` as `patch_rows` (it is patch
  // cols) and `kernel_height` as `patch_cols` (it is patch rows). Also we swap
  // padding top with left and bottom with right.

  // TODO(ezhulenev): Fix dimensions order in Eigen Tensor image patches?
  // clang-format off
  return input
      .extract_image_patches(kernel_width, kernel_height,
                             width_stride, height_stride,
                             width_dilation, height_dilation,
                             width_inflation, heigh_inflation,
                             padding_left, padding_right,
                             padding_top, padding_bottom,
                             /*padding_value=*/static_cast<InputScalar>(0))
      .reshape(pre_contract_dims)
      .contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
      .reshape(post_contract_dims);
  // clang-format on
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_H_
