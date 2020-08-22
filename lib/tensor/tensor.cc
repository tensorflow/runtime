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

//===- tensor.cc ----------------------------------------------------------===//
//
// This file implements Tensor and related types.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/tensor.h"

#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/host_tensor.h"

namespace tfrt {

Tensor::~Tensor() {}

raw_ostream& operator<<(raw_ostream& os, const TensorMetadata& metadata) {
  if (metadata.dtype.kind() == DType::Invalid) {
    os << "<invalid tensor metadata>";
    // Don't print the shape.
    return os;
  }

  return os << metadata.dtype << " " << metadata.shape;
}

AsyncValueRef<HostTensor> Tensor::ConvertToHostTensor(
    HostContext* host, TensorType dst_tensor_type) const {
  return MakeErrorAsyncValueRef(
      host,
      StrCat("Unavailable ConvertToHostTensor with dst_tensor_type_name"));
}

}  // namespace tfrt
