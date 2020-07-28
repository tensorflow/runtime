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

//===- coo_host_tensor_kernels.cc -------------------------------*- c++ -*-===//
//
// This file defines the kernels for COO sparse host tensors.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_tensor_utils.h"

namespace tfrt {

// Returns true if two input tensors are equal, false otherwise.
template <typename T, size_t Rank>
static void SparseTensorEqual(Argument<CooHostTensor> t1,
                              Argument<CooHostTensor> t2, Argument<Chain> chain,
                              Result<bool> output1, Result<Chain> output2) {
  output1.Emplace((MutableDHTArrayView<T>(t1->Values()) ==
                   MutableDHTArrayView<T>(t2->Values())) &&
                  (MutableDHTArrayView<int64_t>(t1->Indices()) ==
                   MutableDHTArrayView<int64_t>(t2->Indices())));
  // Reuse input chain.
  output2.Set(chain);
}

// Converts a sparse tensor in COO layout to a DenseHostTensor.
template <typename T, size_t Rank>
static void ConvertToDHT(Argument<CooHostTensor> in, Argument<Chain> in_chain,
                         Result<DenseHostTensor> out, Result<Chain> out_chain,
                         KernelErrorHandler handler, AsyncKernelFrame* frame) {
  uint32_t allowed_formats =
      1 << static_cast<uint32_t>(Tensor::Subclass::DenseHost);
  auto host_tensor =
      in.get().ConvertToHostTensor(frame->GetHostContext(), allowed_formats);
  auto dht = AsyncValueRef<DenseHostTensor>(host_tensor.ReleaseRCRef());
  out.Set(std::move(dht));
  out_chain.Set(in_chain);
}

// Converts a DenseHostTensor into a CooHostTensor.
// The conversion consists of two passes: One to count how many non-zero
// elements there are in the tensor and another one copy these elements to the
// newly allocated buffer.
template <typename T, size_t Rank>
static void ConvertFromDHT(ArgumentView<MutableDHTIndexableView<T, Rank>> in,
                           Argument<Chain> in_chain, Result<CooHostTensor> out,
                           Result<Chain> out_chain, KernelErrorHandler handler,
                           AsyncKernelFrame* frame) {
  ssize_t num_non_zero_values = 0;
  for (const auto& element : in->Elements()) {
    if (element != 0) {
      num_non_zero_values++;
    }
  }
  auto values = DenseHostTensor::CreateUninitialized<T>(
      TensorShape(num_non_zero_values), frame->GetHostContext());
  if (!values.hasValue()) {
    handler.ReportError("Cannot allocate value tensor");
    return;
  }
  auto indices = DenseHostTensor::CreateUninitialized<int64_t>(
      TensorShape({num_non_zero_values, Rank}), frame->GetHostContext());
  if (!indices.hasValue()) {
    handler.ReportError("Cannot allocate index tensor");
    return;
  }
  auto values_view = MutableDHTIndexableView<T, 1>(values.getPointer());
  auto indices_view = MutableDHTIndexableView<int64_t, 2>(indices.getPointer());
  const auto elements = in->Elements();
  // Index of the next element in the sparse tensor to be filled.
  int sparse_index = 0;
  for (size_t i = 0; i < in->Elements().size(); i++) {
    if (elements[i] != 0) {
      values_view.ElementAt(sparse_index) = elements[i];
      // Make a copy of the index so we can decompose it into its coordinates.
      size_t idx = i;
      // In a row-major layout, the linear index can be calculated as:
      //   linear_index = x_0 + x_1 * dims[0] + x_2 * dims[1] * dims[0] + ...
      // In other words:
      //   linear_index = x_0 + sum_{i=1} (x_i * prod_{j=0}^{i-1} dims[j])
      for (uint64_t r = 0; r < Rank; r++) {
        indices_view.ElementAt(sparse_index, Rank - r - 1) =
            idx % in->FixedShape()[r];
        idx /= in->FixedShape()[r];
      }
      sparse_index++;
    }
  }
  out.Emplace(CooHostTensor(in->FixedShape().ToTensorShape(), GetDType<T>(),
                            std::move(*indices), std::move(*values)));
  out_chain.Set(in_chain);
}

template <typename T, size_t Rank>
void RegisterCooHostTensorKernelsForTypeAndRank(KernelRegistry* registry,
                                                const std::string& t_name) {
  std::string suffix = t_name + "." + std::to_string(Rank);
  registry->AddKernel("coo.convert_coo_to_dht." + suffix,
                      TFRT_KERNEL(ConvertToDHT<T, Rank>));
  registry->AddKernel("coo.tensor_equal." + suffix,
                      TFRT_KERNEL(SparseTensorEqual<T, Rank>));
  registry->AddKernel("coo.convert_dht_to_coo." + suffix,
                      TFRT_KERNEL(ConvertFromDHT<T, Rank>));
}

template <typename T>
void RegisterCooHostTensorKernelsForType(KernelRegistry* registry,
                                         const std::string& t_name) {
  RegisterCooHostTensorKernelsForTypeAndRank<T, 0>(registry, t_name);
  RegisterCooHostTensorKernelsForTypeAndRank<T, 1>(registry, t_name);
  RegisterCooHostTensorKernelsForTypeAndRank<T, 2>(registry, t_name);
  RegisterCooHostTensorKernelsForTypeAndRank<T, 3>(registry, t_name);
}

void RegisterCooHostTensorKernels(KernelRegistry* registry) {
  RegisterCooHostTensorKernelsForType<float>(registry, "f32");
  RegisterCooHostTensorKernelsForType<int32_t>(registry, "i32");
}

}  // namespace tfrt
