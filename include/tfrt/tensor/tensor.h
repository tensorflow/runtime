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

//===- tensor.h -------------------------------------------------*- C++ -*-===//
//
// This file defines the Tensor, TensorLayout, and TensorMetadata types.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_TENSOR_H_
#define TFRT_TENSOR_TENSOR_H_

#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {
class HostContext;
class HostTensor;

// A Tensor is a hyper-rectangular array of values, and the Tensor class is the
// common base class for all tensor implementations in TFRT.
//
// Tensor has a dynamically determined dtype, rank, and tensor type.  It is
// subclassed by more specific types used by various devices.
class Tensor {
 public:
  // TODO(b/149227390): This indicates which subclass of Tensor this dynamically
  // is, allowing efficient downcasting with integer comparisons.  We are likely
  // to make this openly extensible, but for now we just use an enum.
  enum class Subclass {
    DenseHost,   // This is a DenseHostTensor
    ScalarHost,  // This is a ScalarHostTensor
    CooHost,     // This is a CooHostTensor
    StringHost,  // This is a StringHostTensor

    DenseGpu,           // This is a DenseGpuTensor
    TFRuntimeFallback,  // This is a TFRuntimeFallbackTensor
    TFKernelFallback,   // This is a TFKernelFallbackTensor
    TFLiteHost,         // This is a TfLiteHostTensor
    DenseTpu,           // This is a DenseTpuTensor
  };

  DType dtype() const { return metadata_.dtype; }

  // The shape of this tensor.
  const TensorShape& shape() const { return metadata_.shape; }

  const TensorMetadata& metadata() const { return metadata_; }

  TensorType tensor_type() const { return tensor_type_; }

  ssize_t NumElements() const { return shape().GetNumElements(); }

  virtual void Print(raw_ostream& os) const = 0;

  // Copy the contents of this Tensor into HostTensor format and return it as a
  // new tensor.  This can (in general) require device computation or
  // lots of copying, so this returns an AsyncValue for the result.
  //
  // The allowed_formats field is a bitmask indicating supported host formats.
  // it is a bitmask of host Subclass kinds.  DenseHostTensor is always allowed.
  //
  // This returns an error value if the input tensor is invalid or an error is
  // encountered like OOM.
  virtual AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, uint32_t allowed_formats) const = 0;

  // Same as above, except dst_tensor_type_name is used to specify destination
  // tensor type name instead of allowed_formats.
  // TODO(b/163084901): make it a pure virtual function
  virtual AsyncValueRef<HostTensor> ConvertToHostTensor(
      HostContext* host, TensorType dst_tensor_type) const;

  virtual bool IsHostTensor() const { return false; }

  // Note: subclass() exists for implementations of classof(..), which allows
  // dynamic casting with isa<>, dyn_cast<>, etc.  Clients should generally use
  // those templates instead of directly using this member.  You shouldn't
  // switch over this, because your code will have to be updated when new tensor
  // classes get added.
  Subclass subclass() const { return subclass_; }

  bool IsTensorType(TensorType tensor_type) const {
    return this->tensor_type() == tensor_type;
  }

 protected:
  // This class is not copyable or assignable. If we add a copy operation it
  // will likely be explicit - copying a Tensor can be a very expensive
  // operation.
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Tensor(Subclass subclass, const TensorMetadata& metadata)
      : metadata_(metadata), subclass_(subclass) {
    assert(metadata.IsValid() &&
           "Cannot create a tensor with invalid metadata");
  }
  virtual ~Tensor();

  Tensor(Tensor&& other);
  Tensor& operator=(Tensor&& other);

 private:
  TensorMetadata metadata_;
  Subclass subclass_;
  template <class Derived>
  friend class TensorTraits;
  TensorType tensor_type_ = TensorType::kUnknownTensorType;
};

inline Tensor::Tensor(Tensor&& other) = default;
inline Tensor& Tensor::operator=(Tensor&& other) = default;

inline raw_ostream& operator<<(raw_ostream& os, const Tensor& tensor) {
  tensor.Print(os);
  return os;
}

// TensorTraits register TensorType for Derived class. Each sub Tensor needs to
// inherit TensorTraits.
// Example usage:
// class MyTensor : public Tensor, public TensorTraits<MyTensor> {};
template <class Derived>
class TensorTraits {
 public:
  static const tfrt::TensorType& kTensorType;
  TensorTraits() { static_cast<Derived*>(this)->tensor_type_ = kTensorType; }

  static bool classof(const Tensor* t) { return t->IsTensorType(kTensorType); }
};

template <class Derived>
const tfrt::TensorType& TensorTraits<Derived>::kTensorType =
    RegisterStaticTensorType(Derived::name());

}  // namespace tfrt

#endif  // TFRT_TENSOR_TENSOR_H_
