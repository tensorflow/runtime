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

// This file declaress the functions to resize image.

#include "resize_bilinear_op.h"

namespace tfrt {
namespace image {
namespace {

struct CachedInterpolation {
  Index lower;  // Lower source index used in the interpolation
  Index upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

void compute_interpolation_weights(const Index out_size, const Index in_size,
                                   const float scale,
                                   CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (Index i = out_size - 1; i >= 0; --i) {
    const float in = static_cast<float>(i) * scale;
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<Index>(in_f), static_cast<Index>(0));
    interpolation[i].upper =
        std::min(static_cast<Index>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

float compute_lerp(const float top_left, const float top_right,
                   const float bottom_left, const float bottom_right,
                   const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}
}  // namespace

void resize_image(const DenseHostTensor& input, const float height_scale,
                  const float width_scale, DenseHostTensor& output) {
  const TensorShape& input_shape = input.shape();
  Index batch_size = 1;
  Index input_height = input_shape.GetDimensionSize(0);
  Index input_width = input_shape.GetDimensionSize(1);
  Index channels = input_shape.GetDimensionSize(2);

  const TensorShape& output_shape = output.shape();
  Index output_height = output_shape.GetDimensionSize(1);
  Index output_width = output_shape.GetDimensionSize(2);

  std::vector<CachedInterpolation> ys(output_height + 1);
  std::vector<CachedInterpolation> xs(output_width + 1);

  compute_interpolation_weights(output_height, input_height, height_scale,
                                ys.data());
  compute_interpolation_weights(output_width, input_width, width_scale,
                                xs.data());

  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= channels;
    xs[i].upper *= channels;
  }

  const Index in_row_size = input_width * channels;
  const Index in_batch_num_values = input_height * in_row_size;
  const Index out_row_size = output_width * channels;
  const uint8_t* input_b_ptr = static_cast<const uint8_t*>(input.data());
  float* output_y_ptr = static_cast<float*>(output.data());

  for (int b = 0; b < batch_size; ++b) {
    for (Index y = 0; y < output_height; ++y) {
      const uint8_t* ys_input_lower_ptr =
          input_b_ptr + ys[y].lower * in_row_size;
      const uint8_t* ys_input_upper_ptr =
          input_b_ptr + ys[y].upper * in_row_size;
      const float ys_lerp = ys[y].lerp;
      for (Index x = 0; x < output_width; ++x) {
        const Index xs_lower = xs[x].lower;
        const Index xs_upper = xs[x].upper;
        const float xs_lerp = xs[x].lerp;

        // Read channel 0.
        const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
        const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
        const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
        const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

        // Read channel 1.
        const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
        const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
        const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
        const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

        // Read channel 2.
        const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
        const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
        const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
        const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

        // Compute output.
        output_y_ptr[x * channels + 0] =
            compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
                         xs_lerp, ys_lerp);
        output_y_ptr[x * channels + 1] =
            compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
                         xs_lerp, ys_lerp);
        output_y_ptr[x * channels + 2] =
            compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
                         xs_lerp, ys_lerp);
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

}  // namespace image
}  // namespace tfrt
