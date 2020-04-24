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

//===- conv2d_batch_norm_relu.cc ---------------------------------*- C++-*-===//
//
// Conv2D + FusedBatchNorm + Relu kernel registration.
//
//===----------------------------------------------------------------------===//

#include "conv2d.h"

namespace tfrt {

void RegisterConv2DBatchNormReluKernels(KernelRegistry* registry) {
  registry->AddKernel(
      "eigen.conv2d.batch_norm.relu.f32",
      TFRT_KERNEL(compat::internal::Conv2DBatchNorm<float, compat::Relu>));
}

}  // namespace tfrt