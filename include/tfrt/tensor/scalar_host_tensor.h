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

//===- scalar_host_tensor.h -------------------------------------*- C++ -*-===//
//
// This file defines the AnyScalarHostTensor template and ScalarHostTensor
// class.  These represent a scalar value broadcasted to a tensor shape.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_SCALAR_HOST_TENSOR_H_
#define TFRT_TENSOR_SCALAR_HOST_TENSOR_H_

#include "tfrt/tensor/host_tensor.h"

namespace tfrt {

// Represents a tensor whose elements are represented as a broadcasted scalar
// value.
class AnyScalarHostTensor : public HostTensor,
                            public TensorTraits<AnyScalarHostTensor> {
 public:
  AnyScalarHostTensor(TensorMetadata metadata)
      : HostTensor(Subclass::ScalarHost, metadata) {}

  // Return a pointer to the data.
  void* data();
  const void* data() const {
    return const_cast<AnyScalarHostTensor*>(this)->data();
  }

  AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, uint32_t allowed_formats) const override;

  void Print(raw_ostream& os) const override;

  // Tensor type for ScalarHostTensor.
  static const char* name() { return "ScalarHost"; }
};

// This is a ScalarHostTensor of a specific type.  It represents a tensor whose
// elements are represented as a broadcasted scalar value.
template <typename ElementType>
class ScalarHostTensor final : public AnyScalarHostTensor {
 public:
  // Create an uninitialized ScalarHostTensor.
  explicit ScalarHostTensor(const TensorShape& shape)
      : ScalarHostTensor{TensorMetadata{GetDType<ElementType>(), shape}} {}

  explicit ScalarHostTensor(const TensorShape& shape, ElementType value)
      : ScalarHostTensor{TensorMetadata{GetDType<ElementType>(), shape},
                         value} {}
  // Create an uninitialized ScalarHostTensor.
  explicit ScalarHostTensor(TensorMetadata metadata)
      : AnyScalarHostTensor(metadata) {
    assert(metadata.dtype == GetDType<ElementType>());
  }

  // Create an initialized ScalarHostTensor.
  ScalarHostTensor(TensorMetadata metadata, ElementType value)
      : AnyScalarHostTensor(metadata), value_(value) {}

  // Return the element value filling this Tensor.
  const ElementType& GetValue() const { return value_; }
  ElementType& GetValue() { return value_; }

  // Set the element value filling this Tensor.
  void SetValue(ElementType value) { value_ = value; }

  ScalarHostTensor(ScalarHostTensor&& other) = default;
  ScalarHostTensor& operator=(ScalarHostTensor&& other) = default;

  static bool classof(const AnyScalarHostTensor* t) {
    // If the dtype matches, then we're compatible.
    return t->dtype() == GetDType<ElementType>();
  }

  static bool classof(const Tensor* t) {
    auto* t2 = dyn_cast<AnyScalarHostTensor>(t);
    return t2 && classof(t2);
  }

 private:
  ElementType value_;
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_SCALAR_HOST_TENSOR_H_
