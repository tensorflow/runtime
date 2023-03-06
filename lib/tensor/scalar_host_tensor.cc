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

// This file implements ScalarHostTensor.

#include "tfrt/tensor/scalar_host_tensor.h"

#include <optional>

#include "llvm/Support/raw_ostream.h"
#include "tfrt/dtype/dtype_formatter.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/conversion_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

// Return a pointer to the data.
void* AnyScalarHostTensor::data() {
  // This should all constant fold away into something very simple.
  switch (dtype()) {
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
static AsyncValueRef<AnyScalarHostTensor> CopyScalar(
    const ScalarHostTensor<T>& src, HostContext* host) {
  return MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(src.metadata(),
                                                         src.GetValue());
}

void AnyScalarHostTensor::Print(raw_ostream& os) const {
  os << "ScalarHostTensor dtype = " << dtype() << ", shape = " << shape()
     << ", value = " << FormatDType(dtype(), data());
}
static AsyncValueRef<AnyScalarHostTensor>
ConvertScalarHostTensorToScalarHostTensor(const AnyScalarHostTensor& tensor,
                                          const CpuDevice& src,
                                          const CpuDevice& dst,
                                          const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  switch (tensor.dtype()) {
    default:
      llvm_unreachable("can't happen");
#define DTYPE_NUMERIC(ENUM)                                              \
  case DType::ENUM:                                                      \
    return CopyScalar(                                                   \
        *cast<ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(&tensor), \
        host);
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

std::optional<DenseHostTensor> CopyScalarHostTensorToDenseHostTensor(
    const AnyScalarHostTensor& tensor, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto result_alloc =
      DenseHostTensor::CreateUninitialized(tensor.metadata(), host);

  if (!result_alloc) return std::nullopt;

  auto& result_tensor = result_alloc.value();

  auto num_elements = result_tensor.NumElements();
  auto element_size = GetHostSize(tensor.dtype());
  auto* dest_data = result_tensor.data();
  auto* src_data = tensor.data();

  // Fill the DenseHostTensor with the scalar value.  We specialize for a few
  // common sizes here to allow the compiler to specialize for us.
  // TODO(tfrt-devs): This could be done in parallel in the background for
  // large tensors.
  switch (element_size) {
    default:
      // Fully generic size.
      for (Index i = 0; i != num_elements; ++i)
        memcpy(static_cast<int8_t*>(dest_data) + i * element_size, src_data,
               element_size);
      break;
    case sizeof(int8_t):
      memset(dest_data, *static_cast<const unsigned char*>(src_data),
             num_elements);
      break;
    case sizeof(int16_t):
      for (Index i = 0; i != num_elements; ++i)
        *(static_cast<int16_t*>(dest_data) + i) =
            *static_cast<const int16_t*>(src_data);
      break;
    case sizeof(int32_t):
      for (Index i = 0; i != num_elements; ++i)
        *(static_cast<int32_t*>(dest_data) + i) =
            *static_cast<const int32_t*>(src_data);
      break;
    case sizeof(int64_t):
      for (Index i = 0; i != num_elements; ++i)
        *(static_cast<int64_t*>(dest_data) + i) =
            *static_cast<const int64_t*>(src_data);
      break;
  }

  return result_alloc;
}

static AsyncValueRef<DenseHostTensor> ConvertScalarHostTensorToDenseHostTensor(
    const AnyScalarHostTensor& tensor, const CpuDevice& src,
    const CpuDevice& dst, const ExecutionContext& exec_ctx) {
  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>();

  auto optional_dht = CopyScalarHostTensorToDenseHostTensor(tensor, exec_ctx);
  if (!optional_dht)
    return MakeErrorAsyncValueRef("out of memory copying tensor");

  result.emplace(std::move(optional_dht.value()));
  return result;
}

void RegisterScalarHostTensorConversionFn(
    TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertScalarHostTensorToDenseHostTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertScalarHostTensorToScalarHostTensor));
}

}  // namespace tfrt
