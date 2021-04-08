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

// Declares interface for calling hand-written DNN related CUDA kernels.
#ifndef TFRT_BACKENDS_GPU_LIB_OPS_TF_DNN_OPS_CU_H_
#define TFRT_BACKENDS_GPU_LIB_OPS_TF_DNN_OPS_CU_H_

#include "tfrt/common/ops/tf/dnn_ops_util.h"
#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {
class DenseGpuTensor;
class GpuBuffer;

// Transposes `input_filter` according to `channel_order` and writes the
// output into `output_filter` buffer.
// `input_filter` is assumed to be in HWIO layout.
// If `channel_order` is ChannelFirst, does HWIO -> OIHW transform.
// If `channel_order` is ChannelLast, does HWIO -> OHWI transform.
llvm::Error TransformFilterTensor(stream::CurrentContext current,
                                  const stream::Stream& stream,
                                  ChannelOrder channel_order,
                                  const DenseGpuTensor& input_filter,
                                  GpuBuffer* output_filter);

// FusedBatchNormEx op supports side inputs and activations:
//   (1) batch_norm + activation
//   (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode { kIdentity, kRelu };

llvm::Error FusedBatchNormEx(
    stream::CurrentContext current, const stream::Stream& stream,
    ChannelOrder channel_order, const DenseGpuTensor& input,
    const DenseGpuTensor& scale, const DenseGpuTensor& bias,
    const DenseGpuTensor& mean, const DenseGpuTensor& variance,
    const DenseGpuTensor* side_input, float epsilon,
    FusedBatchNormActivationMode activation_mode, GpuBuffer* output_buffer);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_OPS_TF_DNN_OPS_CU_H_
