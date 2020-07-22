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

//===- btf_kernels.cc -----------------------------------------------------===//
//
// This file implements kernels for reading tensors from file.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/btf_reader_util.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace {
template <typename DType_, size_t Rank_>
struct ParseDenseHostTensorTraits {
  using DType = DType_;
  static constexpr size_t kRank = Rank_;
  using TensorTy = DenseHostTensor;
  static constexpr auto kParseTensorFn =
      ParseDenseHostTensorFromStream<DType_, Rank_>;
};

template <typename DType_, size_t Rank_>
constexpr size_t ParseDenseHostTensorTraits<DType_, Rank_>::kRank;

template <size_t Rank>
void RegisterDenseTensorReaders(KernelRegistry* registry) {
  registry->AddKernel(
      "btf.read_dense_tensor.f32." + std::to_string(Rank),
      TFRT_KERNEL(ReadTensorFromBTF<ParseDenseHostTensorTraits<float, Rank>>));
  registry->AddKernel(
      "btf.read_dense_tensor.i32." + std::to_string(Rank),
      TFRT_KERNEL(
          ReadTensorFromBTF<ParseDenseHostTensorTraits<int32_t, Rank>>));
  registry->AddKernel(
      "btf.read_dense_tensor.i64." + std::to_string(Rank),
      TFRT_KERNEL(
          ReadTensorFromBTF<ParseDenseHostTensorTraits<int64_t, Rank>>));
  registry->AddKernel(
      "btf.read_dense_tensor.i8." + std::to_string(Rank),
      TFRT_KERNEL(ReadTensorFromBTF<ParseDenseHostTensorTraits<int8_t, Rank>>));
  registry->AddKernel(
      "btf.read_dense_tensor.ui8." + std::to_string(Rank),
      TFRT_KERNEL(
          ReadTensorFromBTF<ParseDenseHostTensorTraits<uint8_t, Rank>>));
}
}  // namespace

void RegisterBTFIOKernels(KernelRegistry* registry) {
  RegisterDenseTensorReaders<0>(registry);
  RegisterDenseTensorReaders<1>(registry);
  RegisterDenseTensorReaders<2>(registry);
  RegisterDenseTensorReaders<3>(registry);
  RegisterDenseTensorReaders<4>(registry);
}

}  // namespace tfrt
