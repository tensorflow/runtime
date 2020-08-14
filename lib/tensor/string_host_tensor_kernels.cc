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

//===- string_host_tensor_kernels.cc --------------------------------------===//
//
// This file defines the kernels for string host tensors.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/string_host_tensor_kernels.h"

#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/string_host_tensor.h"

namespace tfrt {
namespace {

llvm::Expected<StringHostTensor> CreateStringTensor(
    ArrayAttribute<ssize_t> shape, AggregateAttr values,
    const ExecutionContext& exec_ctx) {
  auto result = StringHostTensor::CreateUninitialized(
      TensorMetadata(DType(DType::String), shape.data()), exec_ctx.host());
  if (!result) {
    return MakeStringError("Failed to create SHT");
  }

  auto strings = result->strings();
  if (strings.size() != values.GetNumElements()) {
    return MakeStringError("Shape mismatch");
  }
  for (int i = 0, e = values.GetNumElements(); i != e; ++i) {
    strings[i] = values.GetAttributeOfType<StringAttr>(i).GetValue().str();
  }

  return std::move(result).getValue();
}

static Expected<StringHostTensor> CreateUninitializedStringTensor(
    ArrayAttribute<ssize_t> shape_in, const ExecutionContext& exec_ctx) {
  auto result = StringHostTensor::CreateUninitialized(
      TensorShape(shape_in.data()), exec_ctx.host());
  if (!result.hasValue()) {
    return MakeStringError("Cannot allocate tensor");
  }
  return std::move(*result);
}

}  // namespace

void RegisterStringHostTensorKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_sht.create_tensor",
                      TFRT_KERNEL(CreateStringTensor));
  registry->AddSyncKernel("tfrt_sht_sync.create_tensor",
                          TFRT_SYNC_KERNEL(CreateStringTensor));
  registry->AddSyncKernel("tfrt_sht_sync.create_uninitialized_tensor",
                          TFRT_SYNC_KERNEL(CreateUninitializedStringTensor));
}

}  // namespace tfrt
