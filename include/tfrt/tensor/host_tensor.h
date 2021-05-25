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

// This file defines the HostTensor class, which is a subclass of Tensor for
// those that live in host memory.

#ifndef TFRT_TENSOR_HOST_TENSOR_H_
#define TFRT_TENSOR_HOST_TENSOR_H_

#include "tfrt/tensor/tensor.h"

namespace tfrt {

// HostTensor is a tensor that lives on the host.
class HostTensor : public Tensor {
 public:
  HostTensor(HostTensor&& other);
  HostTensor& operator=(HostTensor&& other);

  bool IsHostTensor() const override { return true; }

 protected:
  HostTensor() = default;
  explicit HostTensor(const TensorMetadata& metadata) : Tensor(metadata) {}
  ~HostTensor() override = default;
};

inline HostTensor::HostTensor(HostTensor&& other) = default;
inline HostTensor& HostTensor::operator=(HostTensor&& other) = default;

}  // namespace tfrt

#endif  // TFRT_TENSOR_HOST_TENSOR_H_
