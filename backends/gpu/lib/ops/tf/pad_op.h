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

// Declares a function to register tf.Pad op
//
// Declares a function to register tf.Pad implementation on GPU.
#ifndef TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_H_
#define TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_H_

#define EIGEN_USE_GPU

#include "tfrt/common/compat/eigen/tensor_types.h"

namespace tfrt {

class GpuDispatchContext;
class DenseView;
class TensorMetadata;
class OpAttrsRef;

namespace gpu {

class DenseGpuTensor;

namespace functor {

// Functor used by PadOp to do the computations.
template <typename TDevice, typename T, typename Tpadding, int Rank>
struct Pad {
  // Pad "input" into "output", as specified by "paddings" and "pad_value".
  // See pad_op.cc for details.
  void operator()(const TDevice& d, compat::EigenTensor<T, Rank> output,
                  compat::EigenConstTensor<T, Rank> input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, Rank> paddings,
                  T pad_value) {
    // TODO(iga): Not huge paddings can be handled more efficiently. Do
    // necessary eigen magic.
    // if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value &&
    //    (output.size() <= std::numeric_limits<int32_t>::max())) {
    //  To32Bit(output).device(d) = To32Bit(input).pad(paddings, pad_value);
    output.device(d) = input.pad(paddings, pad_value);
  }
};

template <typename TDevice, typename T, typename Tpadding>
struct Pad<TDevice, T, Tpadding, 0> {
  // In the scalar case we simply copy the input.
  void operator()(const TDevice& d, compat::EigenTensor<T, 0> output,
                  compat::EigenConstTensor<T, 0> input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, 0>, T) {
    output.device(d) = input;
  }
};

}  // namespace functor

llvm::Expected<DenseGpuTensor> EnqueueGpuPadOp(GpuDispatchContext* dctx,
                                               const DenseGpuTensor& input,
                                               const DenseView& paddings,
                                               const TensorMetadata& result_md);

}  // namespace gpu

class GpuOpRegistry;

void RegisterPadGpuTfOps(GpuOpRegistry* registry);
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_H_
