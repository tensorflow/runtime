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

// Types for holding dense GPU memory
//
// This file declares classes that can be used to hold dense GPU memory.
#ifndef TFRT_GPU_TENSOR_DENSE_GPU_TENSOR_H_
#define TFRT_GPU_TENSOR_DENSE_GPU_TENSOR_H_

#include <optional>

#include "tfrt/gpu/gpu_types.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {
namespace gpu {

// Tensors are GpuBuffers plus dtype, shape, and layout. CUDA kernels
// that are not specialized to particular input shapes and/or dtypes
// (especially hand-written ones) want to dynamically dispatch
// based on shapes, dtype, and/or layout. These kernels take tensors
// as arguments.
// DenseGpuTensor is thread-safe.
class DenseGpuTensor final : public Tensor,
                             public TensorTraits<DenseGpuTensor> {
 public:
  DenseGpuTensor(const TensorMetadata& metadata,
                 AsyncValueRef<GpuBuffer> buffer);
  DenseGpuTensor(const TensorShape& shape, DType dtype,
                 AsyncValueRef<GpuBuffer> buffer);

  DenseGpuTensor(const DenseGpuTensor& b) = delete;
  DenseGpuTensor& operator=(const DenseGpuTensor& b) = delete;
  DenseGpuTensor(DenseGpuTensor&& b) = default;
  DenseGpuTensor& operator=(DenseGpuTensor&& b) = default;

  size_t DataSizeInBytes() const { return buffer_->size(); }

  DenseGpuTensor CopyRef() const {
    return DenseGpuTensor(metadata(), buffer_.CopyRef());
  }

  // If `new_shape` has the same number of elements as current shape,
  // returns a new DenseGpuTensor that shares the same underlying GpuBuffer and
  // data type as this tensor.
  // Otherwise, returns an empty optional.
  std::optional<DenseGpuTensor> WithShape(const TensorShape& new_shape) const {
    if (new_shape.GetNumElements() != NumElements()) {
      return std::nullopt;
    }
    return DenseGpuTensor(new_shape, dtype(), buffer_.CopyRef());
  }

  // Tensor type for DenseGpuTensor.
  static const char* name() { return "DenseGpu"; }

  void Print(llvm::raw_ostream& os) const override;

  GpuBuffer& buffer() const { return *buffer_; }

  AsyncValueRef<GpuBuffer> CopyBufferRef() const { return buffer_.CopyRef(); }

 private:
  AsyncValueRef<GpuBuffer> buffer_;
};

template <typename T>
T* GetRawPointer(const DenseGpuTensor& tensor) {
  return GetRawPointer<T>(tensor.buffer());
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_TENSOR_DENSE_GPU_TENSOR_H_
