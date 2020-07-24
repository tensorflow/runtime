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

//===- eigen_support.h ------------------------------------------*- C++ -*-===//
//
// This file declares some helpers that tfrt::GpuDevice uses to interact with
// Eigen.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_GPU_LIB_DEVICE_EIGEN_SUPPORT_H_
#define TFRT_BACKENDS_GPU_LIB_DEVICE_EIGEN_SUPPORT_H_

#include <memory>

#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"

namespace Eigen {
class StreamInterface;
struct GpuDevice;
}  // namespace Eigen

namespace tfrt {
namespace gpu {
namespace detail {
// Use custom deleters so we don't need to include Eigen headers.
struct EigenStreamInterfaceDeleter {
  void operator()(::Eigen::StreamInterface* interface) const;
};
struct EigenGpuDeviceDeleter {
  void operator()(::Eigen::GpuDevice* device) const;
};

}  // namespace detail

using OwningEigenStreamInterface =
    std::unique_ptr<::Eigen::StreamInterface,
                    detail::EigenStreamInterfaceDeleter>;
using OwningEigenGpuDevice =
    std::unique_ptr<::Eigen::GpuDevice, detail::EigenGpuDeviceDeleter>;

// Creates and returns an owning handle to an Eigen::StreamInterface instance,
// which wraps the GPU stream 'stream'.
OwningEigenStreamInterface CreateEigenStreamInterface(stream::Stream stream);

// Creates and returns an owning handle to an Eigen::GpuDevice instance,
// which launches GPU kernels on the GPU stream wrapped by 'interface'.
OwningEigenGpuDevice CreateEigenGpuDevice(::Eigen::StreamInterface* interface);

//===----------------------------------------------------------------------===//
// Conversion functions from TFRT GPU tensors to Eigen tensors
//===----------------------------------------------------------------------===//

// TODO(csigg): Should argument be const?
template <typename T, size_t Rank>
compat::EigenTensor<T, Rank> AsEigenTensor(DenseGpuTensor* tensor) {
  assert(tensor->dtype() == GetDType<T>());
  assert(tensor->shape().GetRank() == Rank);
  return compat::EigenTensor<T, Rank>(
      GetRawPointer<T>(*tensor), compat::AsEigenDSizes<Rank>(tensor->shape()));
}

template <typename T, size_t Rank>
compat::EigenConstTensor<T, Rank> AsEigenConstTensor(
    const DenseGpuTensor& tensor) {
  assert(tensor.dtype() == GetDType<T>());
  assert(tensor.shape().GetRank() == Rank);
  return compat::EigenConstTensor<T, Rank>(
      GetRawPointer<T>(tensor), compat::AsEigenDSizes<Rank>(tensor.shape()));
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_DEVICE_EIGEN_SUPPORT_H_
