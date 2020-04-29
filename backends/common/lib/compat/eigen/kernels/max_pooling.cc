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

//===- max_pooling.cc --------------------------------------------*- C++-*-===//
//
// Max Pooling 2D implemented with Eigen.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {

template <typename T>
static void MaxPool2D(ArgumentView<MutableDHTIndexableView<T, 4>> input,
                      ArgumentView<MutableDHTIndexableView<T, 4>> output,
                      Argument<Chain> chain_in, Result<Chain> chain_out,
                      StringAttribute padding,
                      ArrayAttribute<ssize_t> pool_size,
                      ArrayAttribute<ssize_t> strides,
                      KernelErrorHandler handler, HostContext* host,
                      KernelFrame* frame) {
  // TODO(ezhulenev): Move shape computation into support library and share with
  // shape computations in convolution.

  // shape_input has format (batch_size, height, width, channel_num).
  const auto& shape_input = input->FixedShape();
  // shape_output has format (batch_size, height, width, channel_num).
  const auto& shape_output = output->FixedShape();

  if (pool_size.size() != 2) {
    handler.ReportError("MaxPool2D expects pool_size to have 2 elements");
    return;
  }

  if (strides.size() != 2) {
    handler.ReportError("MaxPool2D expects strides to have 2 elements");
    return;
  }

  // Padding for upper, bottom, left and right.
  int padding_numbers[4] = {0, 0, 0, 0};

  if (padding.str() == "same") {
    int total_padding_height = pool_size[0] - strides[0];
    if (shape_input[1] % strides[0] != 0) {
      total_padding_height = pool_size[0] - (shape_input[1] % strides[0]);
    }
    int total_padding_width = pool_size[1] - strides[1];
    if (shape_input[2] % strides[1] != 0) {
      total_padding_width = pool_size[1] - (shape_input[2] % strides[1]);
    }

    padding_numbers[0] = static_cast<int>(total_padding_height / 2.0);
    padding_numbers[1] = static_cast<int>(total_padding_height / 2.0 + 0.5);
    padding_numbers[2] = static_cast<int>(total_padding_width / 2.0);
    padding_numbers[3] = static_cast<int>(total_padding_width / 2.0 + 0.5);
  } else if (padding.str() != "valid") {
    handler.ReportError("MaxPool2D padding '", padding.str(),
                        "' is not recognized");
    return;
  }

  typename MutableDHTIndexableView<T, 4>::FixedShapeType expected_output_shape(
      {shape_input[0],
       (shape_input[1] + padding_numbers[0] + padding_numbers[1] -
        pool_size[0]) /
               strides[0] +
           1,
       (shape_input[2] + padding_numbers[2] + padding_numbers[3] -
        pool_size[1]) /
               strides[1] +
           1,
       shape_input[3]});

  if (shape_output != expected_output_shape) {
    handler.ReportError("MaxPool2D output shape ", shape_output,
                        " does not match the expected output shape ",
                        expected_output_shape);
    return;
  }

  // In the following code we treat every channels vector (innermost dimension)
  // as a single unit for the purpose of computing a maximum value, and pretend
  // that we are working with a tensor of size: [batch, height, width].
  using ChannelVector = Eigen::Tensor<T, 1, Eigen::RowMajor, ssize_t>;
  using InputChannels = Eigen::TensorMap<const ChannelVector, Eigen::Unaligned>;
  using OutputChannels = Eigen::TensorMap<ChannelVector, Eigen::Unaligned>;

  // At every 3 dimensional coordinate we have a vector of size [num_channels].
  const size_t num_channels = shape_output[3];

  // Coordinates: [batch, row, col].
  using Coords = std::array<ssize_t, 3>;

  // Returns OutputChannels for 3 dimensional coordinates.
  const auto output_channels = [output, num_channels](const Coords& coords) {
    T* data = &output->ElementAt(coords[0], coords[1], coords[2], 0);
    return OutputChannels(data, num_channels);
  };

  // Returns InputChannels for 3 dimensional coordinates.
  const auto input_channels = [input, num_channels](const Coords& coords) {
    const T* data = &input->ElementAt(coords[0], coords[1], coords[2], 0);
    return InputChannels(data, num_channels);
  };

  // Strides in the output tensor (excluding innermost channels dimensions).
  const Coords output_strides = {
      shape_output[1] * shape_output[2],  // batch stride
      shape_output[2],                    // height stride
      1                                   // width stride
  };

  // Number of output channel-vectors.
  const ssize_t num_outputs =
      shape_output[0] * shape_output[1] * shape_output[2];

  // Computes [batch, row, col] coordinates of an output channels from one
  // dimensional index in [0, num_outputs) range.
  const auto output_coords = [output_strides](ssize_t index) -> Coords {
    const ssize_t i0 = index / output_strides[0];
    index -= i0 * output_strides[0];

    const ssize_t i1 = index / output_strides[1];
    index -= i1 * output_strides[1];

    const ssize_t i2 = index;

    return {i0, i1, i2};
  };

  // Computes MaxPool outputs in the [start, end) range. All the state captured
  // by value explicitly, because this function will be executed asynchonously.
  auto compute = [pool_size, strides, padding_numbers, shape_input,
                  input_channels, output_channels,
                  output_coords](size_t start, size_t end) -> void {
    // Image patch input channels.
    std::vector<InputChannels> input_channels_pool;
    input_channels_pool.reserve(pool_size[0] * pool_size[1]);

    // Iterate over all outputs in the [start, end) range.
    for (ssize_t index = start; index < end; ++index) {
      const Coords coords = output_coords(index);
      input_channels_pool.clear();

      // Iterate over the spatial pooling patch.
      for (ssize_t x = 0; x < pool_size[0]; ++x) {
        for (ssize_t y = 0; y < pool_size[1]; ++y) {
          // Coordinates in the input tensor.
          const Coords input_coords = {
              coords[0],                                        // batch
              coords[1] * strides[0] + x - padding_numbers[0],  // row (height)
              coords[2] * strides[1] + y - padding_numbers[2],  // col (width)
          };

          // Check if the input coordinates are in the padded space.
          const bool pad =
              input_coords[1] < 0 || input_coords[1] >= shape_input[1] ||
              input_coords[2] < 0 || input_coords[2] >= shape_input[2];
          if (!pad) {
            input_channels_pool.push_back(input_channels(input_coords));
          }
        }
      }

      assert(!input_channels_pool.empty());
      ssize_t i = 0;

      // Initialize output channels.
      auto out = output_channels(coords);
      out = out.constant(std::numeric_limits<T>::min());

      // Process 3 input channels in a single Eigen expression to minimize
      // memory traffic and keep temporary data in the registers.
      const ssize_t vectorized_pooling =
          static_cast<ssize_t>(input_channels_pool.size()) - 3;
      for (; i < vectorized_pooling; i += 3) {
        auto in0 = input_channels_pool[i + 0];
        auto in1 = input_channels_pool[i + 1];
        auto in2 = input_channels_pool[i + 2];
        out = out.cwiseMax(in0).cwiseMax(in1.cwiseMax(in2));
      }

      // Process remaining channels one by one.
      for (; i < static_cast<ssize_t>(input_channels_pool.size()); ++i) {
        auto in0 = input_channels_pool[i];
        out = out.cwiseMax(in0);
      }
    }
  };

  // Compute minimum parallel for block size, to make sure that we do not create
  // too many small tasks if extracted image patches are tiny.
  // TODO(ezhulenev): Use Eigen expression cost model? Or add TFRT cost model?
  static constexpr size_t kMinPatchSize = 1000;
  const size_t image_patch_size = num_channels * pool_size[0] * pool_size[1];
  const size_t min_block_size =
      std::max(static_cast<size_t>(1), kMinPatchSize / image_patch_size);

  host->ParallelFor(
      num_outputs, std::move(compute),
      [chain = chain_out.Allocate(), frame = RAIIKernelFrame(*frame)]() {
        chain.emplace();
      },
      min_block_size);
}

}  // namespace compat

void RegisterMaxPoolingKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.max_pooling_2d.f32",
                      TFRT_KERNEL(compat::MaxPool2D<float>));
}

}  // namespace tfrt
