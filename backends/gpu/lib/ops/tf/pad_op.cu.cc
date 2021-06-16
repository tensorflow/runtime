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

//===- pad_op.cu.cc - Instantiates tf.Pad GPU functors ---------*- C++ -*--===//
//
// Instantiates templated functors used in implementation of tf.Pad op on GPU.
#include "pad_op.h"

#include <iostream>  // some eigen header use std::cerr without including it.

#include "blas_support.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tfrt {
namespace gpu {

// Instantiate templates defined in pad_op.h.
// We need to instantiate them here because they need to be compiled with nvcc.
#define DEFINE_GPU_PAD_SPECS(T, Tpadding)                         \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 0>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 1>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 2>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 3>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 4>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 5>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 6>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 7>; \
  template struct functor::Pad<Eigen::GpuDevice, T, Tpadding, 8>;

#define DEFINE_GPU_SPECS(T)        \
  DEFINE_GPU_PAD_SPECS(T, int32_t) \
  DEFINE_GPU_PAD_SPECS(T, int64_t)

#define DTYPE_NUMERIC(ENUM) \
  DEFINE_GPU_SPECS(EigenTypeForDTypeKind<DType::ENUM>);
#include "tfrt/dtype/dtype.def"

}  // namespace gpu

}  // namespace tfrt
