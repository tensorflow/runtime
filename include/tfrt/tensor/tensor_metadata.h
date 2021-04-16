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

// This file defines the TensorMetadata struct.

#ifndef TFRT_TENSOR_TENSORMETADATA_H_
#define TFRT_TENSOR_TENSORMETADATA_H_

#include "tfrt/dtype/dtype.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// The metadata of a rectangular tensor that can be computed by a metadata
// function.
struct TensorMetadata {
  TensorMetadata() : shape({}), dtype() {}

  TensorMetadata(DType dtype, const TensorShape& shape)
      : shape(shape), dtype(dtype) {}

  template <typename T = int64_t>
  TensorMetadata(DType dtype, ArrayRef<T> shape) : shape(shape), dtype(dtype) {}

  template <typename Container>
  TensorMetadata(DType dtype, const Container& shape)
      : TensorMetadata(dtype, llvm::makeArrayRef(shape)) {}

  bool IsValid() const { return dtype.IsValid(); }
  bool IsInvalid() const { return dtype.IsInvalid(); }

  size_t GetHostSizeInBytes() const {
    return dtype.GetHostSize() * shape.GetNumElements();
  }

  TensorShape shape;
  DType dtype;
};

inline bool operator==(const TensorMetadata& lhs, const TensorMetadata& rhs) {
  return lhs.dtype == rhs.dtype && lhs.shape == rhs.shape;
}

inline bool operator!=(const TensorMetadata& lhs, const TensorMetadata& rhs) {
  return !(lhs == rhs);
}

raw_ostream& operator<<(raw_ostream& os, const TensorMetadata& metadata);

}  // namespace tfrt

#endif  // TFRT_TENSOR_TENSORMETADATA_H_
