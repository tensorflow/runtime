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

//===- zero_padding.cc -------------------------------------------*- C++-*-===//
//
// Pad input tensor with zeroes.
//
//===----------------------------------------------------------------------===//

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace compat {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;

template <typename T>
static void ZeroPadding(ArgumentView<DHTIndexableView<T, 4>> input,
                        ArgumentView<MutableDHTIndexableView<T, 4>> output,
                        Argument<Chain> chain_in, Result<Chain> chain_out,
                        ArrayAttribute<ssize_t> padding,
                        KernelErrorHandler handler,
                        const ExecutionContext& exec_ctx, KernelFrame* frame) {
  // input_shape has format (batch_size, height, width, in_channel_num).
  const auto& input_shape = input->FixedShape();
  const auto& output_shape = output->FixedShape();

  if (padding.size() != 2) {
    handler.ReportError("ZeroPadding expects padding of length 2");
    return;
  }

  // output_shape has format (batch_size, height, width, out_channel_num).
  const FixedRankShape<4> expected_output_shape(
      {input_shape[0], input_shape[1] + 2 * padding[0],
       input_shape[2] + 2 * padding[1], input_shape[3]});

  if (output_shape != expected_output_shape) {
    handler.ReportError("ZeroPadding output shape ", output_shape,
                        " does not match the expected output shape ",
                        expected_output_shape);
    return;
  }

  auto input_t = AsEigenConstTensor(input.get());
  auto output_t = AsEigenTensor(output.get());

  const std::pair<Index, Index> pad_height = {padding[0], padding[0]};
  const std::pair<Index, Index> pad_width = {padding[1], padding[1]};

  const Eigen::array<std::pair<Index, Index>, 4> paddings = {
      {0, 0}, pad_height, pad_width, {0, 0}};

  AsyncAssign(exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
              std::move(output_t), input_t.pad(paddings),
              [chain = chain_out.Allocate(),
               frame = RAIIKernelFrame(*frame)]() { chain.emplace(); });
}

}  // namespace compat

void RegisterZeroPaddingKernels(KernelRegistry* registry) {
  registry->AddKernel("eigen.zero_padding.f32",
                      TFRT_KERNEL(compat::ZeroPadding<float>));
}

}  // namespace tfrt
