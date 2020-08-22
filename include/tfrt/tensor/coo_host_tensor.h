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

//===- coo_host_tensor.h ----------------------------------------*- C++ -*-===//
//
// This file define the CooHostTensor class.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_COO_HOST_TENSOR_H_
#define TFRT_TENSOR_COO_HOST_TENSOR_H_
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

class TensorConversionFnRegistry;

void RegisterCooHostTensorKernels(KernelRegistry* registry);
void RegisterCooHostTensorConversionFn(TensorConversionFnRegistry* registry);

// Represents a sparse tensor as a coordinate list (COO).
class CooHostTensor final : public HostTensor,
                            public TensorTraits<CooHostTensor> {
 public:
  // Empty and null by default.
  CooHostTensor() = default;

  CooHostTensor(const TensorShape& shape, DType dtype,
                DenseHostTensor&& indices, DenseHostTensor&& values)
      : HostTensor(Subclass::CooHost, TensorMetadata(dtype, shape)),
        indices_(std::move(indices)),
        values_(std::move(values)) {}

  // Raw access to data.
  const DenseHostTensor* Values() const { return &values_; }
  DenseHostTensor* Values() { return &values_; }
  const DenseHostTensor* Indices() const { return &indices_; }
  DenseHostTensor* Indices() { return &indices_; }

  void Print(raw_ostream& os) const override;

  AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, uint32_t allowed_formats) const override;

  AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, TensorType dst_tensor_type_id) const override;

  // Tensor type for CooHostTensor.
  static const char* name() { return "CooHost"; }

 private:
  DenseHostTensor indices_;
  DenseHostTensor values_;
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_COO_HOST_TENSOR_H_
