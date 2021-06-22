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

// This file implements the CooHostTensor class.

#include "tfrt/tensor/coo_host_tensor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/dtype/dtype_formatter.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/conversion_utils.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"

namespace tfrt {

namespace {
template <typename DType>
void ConvertToDHTTensorHelper(const DenseHostTensor &indices,
                              const DenseHostTensor &values,
                              DenseHostTensor *result_tensor) {
  auto result_tensor_view = MutableDHTArrayView<DType>(result_tensor);
  const TensorMetadata &result_metadata = result_tensor->metadata();
  const auto &result_shape = result_metadata.shape;
  result_tensor_view.Fill(DType(0));
  auto indices_view = DHTIndexableView<int64_t, 2>(&indices);
  auto values_view = DHTIndexableView<DType, 1>(&values);
  for (int i = 0, e = values_view.FixedShape().GetNumElements(); i != e; ++i) {
    size_t offset = 0;
    size_t stride = 1;
    for (int j = result_shape.GetRank() - 1; j >= 0; --j) {
      assert(indices_view.ElementAt(i, j) < result_shape.GetDimensionSize(j));
      offset += stride * indices_view.ElementAt(i, j);
      stride *= result_shape.GetDimensionSize(j);
    }
    result_tensor_view[offset] = values_view.ElementAt(i);
  }
}
}  // namespace

void CooHostTensor::Print(raw_ostream &os) const {
  // Just dumps the flat values for now.
  os << "CooHostTensor dtype = " << dtype() << ", shape = " << shape();
  os << ", indices = [";

  llvm::interleaveComma(DHTIndexableView<int64_t, 2>(Indices()).Elements(), os);
  os << "], values = [";

  auto element_size = GetHostSize(dtype());
  auto *data_ptr = static_cast<const char *>(Values()->data());
  for (ssize_t i = 0, e = Values()->NumElements(); i != e; ++i) {
    if (i != 0) os << ", ";
    os << FormatDType(dtype(), data_ptr + i * element_size);
  }
  os << "]\n";
}

static AsyncValueRef<AnyScalarHostTensor>
ConvertCooHostTensorToScalarHostTensor(const CooHostTensor &coo,
                                       const CpuDevice &src,
                                       const CpuDevice &dst,
                                       const ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();
  // Allows conversion to ScalarHostTensor if at most one element or if it is an
  // arbitrary-shaped COO tensor but all elements are zero.
  switch (coo.dtype()) {
    default:
      llvm_unreachable("can't happen");
#define DTYPE_NUMERIC(ENUM)                                                 \
  case DType::ENUM:                                                         \
    if (coo.NumElements() == 0) {                                           \
      return MakeAvailableAsyncValueRef<                                    \
          ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(host,            \
                                                           coo.metadata()); \
    } else if (coo.NumElements() == 1) {                                    \
      return MakeAvailableAsyncValueRef<                                    \
          ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(                 \
          host, coo.metadata(),                                             \
          DHTArrayView<TypeForDTypeKind<DType::ENUM>>(coo.Values())[0]);    \
    } else if (coo.Indices()->NumElements() == 0) {                         \
      return MakeAvailableAsyncValueRef<                                    \
          ScalarHostTensor<TypeForDTypeKind<DType::ENUM>>>(                 \
          host, coo.metadata(), TypeForDTypeKind<DType::ENUM>(0));          \
    }
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return MakeErrorAsyncValueRef(
      host, StrCat("failed to convert coo tensor to scalar host tensor"));
}

static AsyncValueRef<DenseHostTensor> ConvertCooHostTensorToDenseHostTensor(
    const CooHostTensor &tensor, const CpuDevice &src, const CpuDevice &dst,
    const ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();
  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);
  auto result_alloc =
      DenseHostTensor::CreateUninitialized(tensor.metadata(), host);
  if (!result_alloc) {
    return MakeErrorAsyncValueRef(
        host, "out of memory converting coo tensor to dht tensor");
  }
  auto &result_tensor = result_alloc.getValue();

  switch (tensor.dtype()) {
    default:
      llvm_unreachable("can't happen");
#define DTYPE_NUMERIC(ENUM)                                   \
  case DType::ENUM:                                           \
    ConvertToDHTTensorHelper<TypeForDTypeKind<DType::ENUM>>(  \
        *tensor.Indices(), *tensor.Values(), &result_tensor); \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
  result.emplace(std::move(result_tensor));
  return result;
}

void RegisterCooHostTensorConversionFn(TensorConversionFnRegistry *registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertCooHostTensorToDenseHostTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertCooHostTensorToScalarHostTensor));
}

}  // namespace tfrt
