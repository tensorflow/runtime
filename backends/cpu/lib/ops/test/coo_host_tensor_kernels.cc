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

// This file implements kernels for handling COO host tensors.

#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/btf.h"
#include "tfrt/tensor/btf_util.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

template <typename DType, size_t Rank>
Expected<CooHostTensor> ParseCooTensorFromStream(std::ifstream* stream,
                                                 size_t offset,
                                                 btf::TensorLayout layout,
                                                 HostContext* host) {
  if (layout != btf::TensorLayout::kCOO_EXPERIMENTAL) {
    return MakeStringError("unexpected tensor layout ", layout);
  }

  std::array<ssize_t, Rank> dims;
  if (!ReadStream(stream, dims.data(), Rank)) {
    return MakeStringError("failed to read tensor dims at offset ", offset);
  }

  auto dht =
      DenseHostTensor::CreateUninitialized<DType>(TensorShape(dims), host);
  if (!dht.hasValue()) {
    return MakeStringError("cannot allocate result tensor");
  }

  Expected<DenseHostTensor> indices =
      ParseDenseHostTensorFromStream<int64_t, 2>(stream, stream->tellg(),
                                                 btf::TensorLayout::kRMD, host);
  if (!indices) {
    return indices.takeError();
  }

  Expected<DenseHostTensor> values = ParseDenseHostTensorFromStream<DType, 1>(
      stream, stream->tellg(), btf::TensorLayout::kRMD, host);
  if (!values) {
    return indices.takeError();
  }

  // Validate COO-specific constraints.
  if (indices.get().NumElements() != values.get().NumElements() * Rank) {
    return MakeStringError(
        "the tensor is not corectly formatted, the indices and "
        "values tensors do not match in size ",
        layout);
  }

  FixedRankShape<Rank> shape(dims);

  // Ensure that the indices are smaller than their respective dimensions.
  auto indices_view = MutableDHTIndexableView<int64_t, 2>(&indices.get());
  for (size_t i = 0; i < indices_view.FixedShape()[0]; i++) {
    for (size_t j = 0; j < indices_view.FixedShape()[1]; j++) {
      if (indices_view.ElementAt(i, j) >= shape[j]) {
        return MakeStringError(
            "the indices tensor has an element at position ", i, ", ", j,
            " that bigger than the respective dimension. Element: ",
            indices_view.ElementAt(i, j), ", limit: ", shape[j]);
      }
    }
  }

  auto sparse_tensor =
      CooHostTensor(shape.ToTensorShape(), GetDType<DType>(),
                    std::move(indices.get()), std::move(values.get()));
  return std::move(sparse_tensor);
}

namespace {

template <typename DType_, size_t Rank_>
struct ParseCooTensorTraits {
  using DType = DType_;
  static constexpr size_t kRank = Rank_;
  using TensorTy = CooHostTensor;
  static constexpr auto kParseTensorFn =
      ParseCooTensorFromStream<DType_, Rank_>;
};

template <typename DType_, size_t Rank_>
constexpr size_t ParseCooTensorTraits<DType_, Rank_>::kRank;

template <size_t N>
void RegisterCooTensorReaders(KernelRegistry* registry) {
  registry->AddKernel(
      "btf.read_coo_tensor.f32." + std::to_string(N),
      TFRT_KERNEL(ReadTensorFromBTF<ParseCooTensorTraits<float, N>>));
  registry->AddKernel(
      "btf.read_coo_tensor.i32." + std::to_string(N),
      TFRT_KERNEL(ReadTensorFromBTF<ParseCooTensorTraits<int32_t, N>>));
  registry->AddKernel(
      "btf.read_coo_tensor.i8." + std::to_string(N),
      TFRT_KERNEL(ReadTensorFromBTF<ParseCooTensorTraits<int8_t, N>>));
}

static CooHostTensor CreateCooTensorOp(const DenseHostTensor& indices,
                                       const DenseHostTensor& values,
                                       const TensorMetadata& dest_md) {
  return CooHostTensor(dest_md.shape, dest_md.dtype, indices.CopyRef(),
                       values.CopyRef());
}
}  // namespace

void RegisterCooKernels(KernelRegistry* registry) {
  RegisterCooTensorReaders<0>(registry);
  RegisterCooTensorReaders<1>(registry);
  RegisterCooTensorReaders<2>(registry);
  RegisterCooTensorReaders<3>(registry);
  RegisterCooTensorReaders<4>(registry);
}

void RegisterCooCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tfrt_test.create_coo_tensor",
                     TFRT_CPU_OP(CreateCooTensorOp), CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
