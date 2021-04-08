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

// This file implements the functions to resize image.

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_

#include "jpeg/jpeg_mem.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"

namespace tfrt {
namespace image {

void resize_image(const DenseHostTensor& input, const float height_scale,
                  const float width_scale, DenseHostTensor& output);

}  // namespace image
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
