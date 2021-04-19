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

//===- dense_gpu_tensor.cc - CUDA tensor implementation -------------------===//
//
// This file implements the DenseGpuTensor class.

#include "tfrt/gpu/tensor/dense_gpu_tensor.h"

#include "llvm/Support/ErrorHandling.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace gpu {

DenseGpuTensor::DenseGpuTensor(const TensorShape& shape, DType dtype,
                               RCReference<GpuCrtBuffer> buffer)
    : DenseGpuTensor(TensorMetadata(dtype, shape), std::move(buffer)) {}

DenseGpuTensor::DenseGpuTensor(const TensorMetadata& metadata,
                               RCReference<GpuCrtBuffer> buffer)
    : Tensor(metadata), buffer_(std::move(buffer)) {}

void DenseGpuTensor::Print(llvm::raw_ostream& os) const {
  os << "DenseGpuTensor<dtype=" << dtype() << ", shape=" << shape()
     << ", pointer=" << buffer_->pointer() << ">";
}

}  // namespace gpu
}  // namespace tfrt
