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

//===- scalar_host_tensor.cc ----------------------------------------------===//
//
// This file implements ScalarHostTensor.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/scalar_host_tensor.h"

#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

// Return a pointer to the data.
void* AnyScalarHostTensor::data() {
  // This should all constant fold away into something very simple.
  switch (dtype().kind()) {
    default:
      llvm_unreachable("can't happen");
#define DTYPE_NUMERIC(ENUM)                                             \
  case DType::ENUM:                                                     \
    return &cast<ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(this) \
                ->GetValue();
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

template <typename T>
static AsyncValueRef<HostTensor> CopyScalar(const ScalarHostTensor<T>& src,
                                            HostContext* host) {
  return MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(host, src.metadata(),
                                                         src.GetValue());
}

AsyncValueRef<HostTensor> AnyScalarHostTensor::ConvertToHostTensor(
    HostContext* host, uint32_t allowed_formats) const {
  // If the caller allows ScalarHostTensor, then we can stay compact.  This
  // still requires a copy of the data though.
  if (allowed_formats &
      (1 << static_cast<uint32_t>(Tensor::Subclass::ScalarHost))) {
    switch (dtype().kind()) {
      default:
        llvm_unreachable("can't happen");
#define DTYPE_NUMERIC(ENUM) \
  case DType::ENUM:         \
    return CopyScalar(      \
        *cast<ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(this), host);
#include "tfrt/dtype/dtype.def"  // NOLINT
    }
  }

  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);

  auto result_alloc = DenseHostTensor::CreateUninitialized(metadata(), host);
  if (!result_alloc)
    return MakeErrorAsyncValueRef(host, "out of memory copying tensor");

  auto& result_tensor = result_alloc.getValue();

  auto num_elements = result_tensor.NumElements();
  auto element_size = dtype().GetHostSize();
  auto* dest_data = result_tensor.data();
  auto* src_data = data();

  // Fill the DenseHostTensor with the scalar value.  We specialize for a few
  // common sizes here to allow the compiler to specialize for us.
  // TODO(tfrt-devs): This could be done in parallel in the background for
  // large tensors.
  switch (element_size) {
    default:
      // Fully generic size.
      for (ssize_t i = 0; i != num_elements; ++i)
        memcpy(static_cast<int8_t*>(dest_data) + i * element_size, src_data,
               element_size);
      break;
    case sizeof(int8_t):
      memset(dest_data, *static_cast<const unsigned char*>(src_data),
             num_elements);
      break;
    case sizeof(int16_t):
      for (ssize_t i = 0; i != num_elements; ++i)
        *(static_cast<int16_t*>(dest_data) + i) =
            *static_cast<const int16_t*>(src_data);
      break;
    case sizeof(int32_t):
      for (ssize_t i = 0; i != num_elements; ++i)
        *(static_cast<int32_t*>(dest_data) + i) =
            *static_cast<const int32_t*>(src_data);
      break;
    case sizeof(int64_t):
      for (ssize_t i = 0; i != num_elements; ++i)
        *(static_cast<int64_t*>(dest_data) + i) =
            *static_cast<const int64_t*>(src_data);
      break;
  }

  result.emplace(std::move(result_tensor));
  return result;
}

void AnyScalarHostTensor::Print(raw_ostream& os) const {
  os << "ScalarHostTensor dtype = " << dtype() << ", shape = " << shape()
     << ", value = ";
  dtype().Print(data(), os);
}

}  // namespace tfrt
