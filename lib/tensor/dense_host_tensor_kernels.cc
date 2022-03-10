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

// This file defines the kernels for dense host tensors.

#include "tfrt/tensor/dense_host_tensor_kernels.h"

#include <complex>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

template <typename T, size_t Rank>
static Expected<DenseHostTensor> CreateUninitializedDenseTensor(
    ArrayAttribute<Index> shape_in, const ExecutionContext& exec_ctx) {
  auto result = DenseHostTensor::CreateUninitialized<T>(
      TensorShape(shape_in.data()), exec_ctx.host());
  if (!result.hasValue()) {
    return MakeStringError("Cannot allocate tensor");
  }
  return std::move(*result);
}

template <typename T>
static void MakeTensor(Argument<RCReference<HostBuffer>> buffer,
                       Argument<TensorShape> shape, Argument<Chain> in_chain,
                       Result<DenseHostTensor> tensor, Result<Chain> out_chain,
                       KernelErrorHandler handler) {
  if (buffer.get()->size() !=
      shape->GetNumElements() * GetHostSize(GetDType<T>())) {
    std::string error_msg;
    llvm::raw_string_ostream ss(error_msg);
    ss << "dht.make_tensor failed: buffer_size (" << buffer.get()->size()
       << ") is not equal to the number of elements in shape (" << *shape
       << ") times element size (" << GetHostSize(GetDType<T>()) << ")";
    handler.ReportError(ss.str());
    return;
  }
  tensor.Emplace(TensorMetadata(GetDType<T>(), *shape), std::move(*buffer));
  out_chain.Set(in_chain);
}

// Constructs a `DenseHostTensor` from the given host buffer and shape.
template <typename T>
static Expected<DenseHostTensor> SyncMakeTensor(
    const RCReference<HostBuffer>& buffer, TensorShape shape) {
  if (buffer->size() != shape.GetNumElements() * GetHostSize(GetDType<T>())) {
    std::string error_msg;
    llvm::raw_string_ostream ss(error_msg);
    ss << "tfrt_dht_sync.make_tensor failed: buffer_size (" << buffer->size()
       << ") is not equal to the number of elements in shape (" << shape
       << ") times element size (" << GetHostSize(GetDType<T>()) << ")";
    return MakeStringError(ss.str());
  }
  return DenseHostTensor(TensorMetadata(GetDType<T>(), shape), buffer);
}

template <typename T>
static Expected<DenseHostTensor> CreateDenseTensor(
    ArrayAttribute<Index> shape, ArrayAttribute<T> values,
    const ExecutionContext& exec_ctx) {
  auto result = DenseHostTensor::CreateUninitialized<T>(
      TensorShape(shape.data()), exec_ctx.host());
  if (!result.hasValue()) {
    return MakeStringError("Cannot allocate tensor");
  }

  MutableDHTArrayView<T> dst{&*result};
  if (values.size() == 1) {
    dst.Fill(values[0]);
  } else {
    assert(values.size() == dst.NumElements());
    std::copy(values.data().begin(), values.data().end(),
              dst.Elements().begin());
  }

  return std::move(*result);
}

template <typename T>
static ScalarHostTensor<T> CreateFromScalar(ArrayAttribute<Index> shape,
                                            Attribute<T> value) {
  return ScalarHostTensor<T>{TensorShape{shape.data()}, *value};
}

template <typename T>
static Chain FillDenseTensorWithConstantValue(
    ArgumentView<MutableDHTArrayView<T>> in, Attribute<T> value) {
  in->Fill(*value);
  return Chain();
}

template <typename T>
static void SetDenseTensorWithConstantValues(
    ArgumentView<MutableDHTArrayView<T>> in, Argument<Chain> chain_in,
    Result<Chain> chain_out, ArrayAttribute<T> values,
    KernelErrorHandler handler) {
  if (in->NumElements() != values.size()) {
    handler.ReportError("Incorrect number of values for the tensor: ",
                        values.size(), ", but expected ", in->NumElements());
    return;
  }
  std::copy(values.data().begin(), values.data().end(), in->Elements().begin());
  chain_out.Set(chain_in);
}

template <typename T>
static Error SyncSetDenseTensorWithConstantValues(MutableDHTArrayView<T> in,
                                                  ArrayAttribute<T> values) {
  if (in.NumElements() != values.size()) {
    return MakeStringError("Incorrect number of values for the tensor: ",
                           values.size(), ", but expected ", in.NumElements());
  }
  std::copy(values.data().begin(), values.data().end(), in.Elements().begin());
  return Error::success();
}

template <typename T>
static llvm::Expected<Chain> SetDenseTensorWithValues(
    MutableDHTArrayView<T> input, const Chain& chain_in,
    RemainingArguments values) {
  if (input.NumElements() != values.size()) {
    return MakeStringError(
        "Incorrect number of values for the tensor: ", values.size(),
        ", but expected ", input.NumElements());
  }

  T* data = input.data();
  for (int i = 0, e = values.size(); i < e; i++) {
    data[i] = values[i]->get<T>();
  }
  return Chain();
}

static Chain NoOpHostTensor(Argument<DenseHostTensor> in) { return Chain(); }

static llvm::Expected<RCReference<HostBuffer>> AllocateBuffer(
    int64_t size, int64_t alignment, AsyncKernelFrame* frame) {
  auto data = HostBuffer::CreateUninitialized(
      static_cast<size_t>(size), static_cast<size_t>(alignment),
      frame->GetHostContext()->allocator());
  if (!data) return MakeStringError("Cannot allocate host buffer");
  return std::move(data);
}

// Constructs a `HostBuffer` based on the given size (in bytes) and alignment.
static llvm::Expected<RCReference<HostBuffer>> SyncAllocateBuffer(
    int64_t size, int64_t alignment, SyncKernelFrame* frame) {
  auto data = HostBuffer::CreateUninitialized(
      static_cast<size_t>(size), static_cast<size_t>(alignment),
      frame->GetHostContext()->allocator());
  if (!data) return MakeStringError("Cannot allocate host buffer");
  return std::move(data);
}

static Chain PrintTensor(const Tensor& t) {
  tfrt::outs() << t << "\n";
  tfrt::outs().flush();
  return Chain();
}

static void SyncPrintTensor(const Tensor& t) {
  tfrt::outs() << t << "\n";
  tfrt::outs().flush();
}

static Chain PrintDenseTensorShape(Argument<DenseHostTensor> t) {
  tfrt::outs() << t->shape() << "\n";
  tfrt::outs().flush();
  return Chain();
}

static TensorShape GetDenseTensorShape(const DenseHostTensor& t) {
  return t.shape();
}

template <typename T>
static void DenseTensorEqual(Argument<DenseHostTensor> t1,
                             Argument<DenseHostTensor> t2,
                             Argument<Chain> chain, Result<bool> output1,
                             Result<Chain> output2) {
  output1.Emplace(TensorEqual<T>(*t1, *t2));
  // Reuse input chain.
  output2.Set(chain);
}

// Returns true if two input tensors are near enough to each other.
template <typename T, int ULP = 2>
static void DenseTensorAllClose(Argument<DenseHostTensor> t1,
                                Argument<DenseHostTensor> t2,
                                Argument<Chain> chain, Result<bool> output1,
                                Result<Chain> output2) {
  output1.Emplace(TensorApproxEqual<T, ULP>(*t1, *t2));
  // Reuse input chain.
  output2.Set(chain);
}

static void GetBuffer(Argument<DenseHostTensor> t, Argument<Chain> chain_in,
                      Result<RCReference<HostBuffer>> buffer,
                      Result<Chain> chain_out) {
  buffer.Emplace(t->buffer());
  chain_out.Set(chain_in);
}

static llvm::Expected<RCReference<HostBuffer>> GetBufferSlice(
    const RCReference<HostBuffer>& parent_buffer, int64_t offset,
    int64_t size) {
  auto data = tfrt::HostBuffer::CreateFromExternal(
      parent_buffer, static_cast<size_t>(offset), static_cast<size_t>(size));
  if (!data) return MakeStringError("Cannot allocate host buffer.");
  return std::move(data);
}

static void PrintBuffer(Argument<RCReference<HostBuffer>> buffer,
                        Argument<Chain> chain_in, Result<Chain> chain_out) {
  tfrt::outs() << **buffer << "\n";
  tfrt::outs().flush();
  chain_out.Set(chain_in);
}

template <typename T, size_t Rank>
static void RegisterDhtCreationKernelsForTypeAndRank(
    KernelRegistry* registry, const std::string& t_name) {
  std::string suffix = t_name + "." + std::to_string(Rank);
  registry->AddKernel("tfrt_dht.create_uninitialized_tensor." + suffix,
                      TFRT_KERNEL(CreateUninitializedDenseTensor<T, Rank>));
  registry->AddSyncKernel(
      "tfrt_dht_sync.create_uninitialized_tensor." + suffix,
      TFRT_SYNC_KERNEL(CreateUninitializedDenseTensor<T, Rank>));
}

template <typename T>
static void RegisterDhtCreationKernelsForType(KernelRegistry* registry,
                                              const std::string& t_name) {
  std::string suffix = t_name;
  registry->AddSyncKernel("tfrt_dht_sync.create_dense_tensor." + suffix,
                          TFRT_SYNC_KERNEL(CreateDenseTensor<T>));
  registry->AddSyncKernel("tfrt_dht_sync.create_from_scalar." + suffix,
                          TFRT_SYNC_KERNEL(CreateFromScalar<T>));
  // Constant is in the name because we will presumably want a version that
  // uses a variable.
  registry->AddKernel("tfrt_dht.fill_tensor_with_constant." + suffix,
                      TFRT_KERNEL(FillDenseTensorWithConstantValue<T>));
  registry->AddKernel("tfrt_dht.make_tensor." + suffix,
                      TFRT_KERNEL(MakeTensor<T>));
  registry->AddSyncKernel("tfrt_dht_sync.make_tensor." + suffix,
                          TFRT_SYNC_KERNEL(SyncMakeTensor<T>));
  registry->AddKernel("tfrt_dht.set_tensor_with_constant_values." + suffix,
                      TFRT_KERNEL(SetDenseTensorWithConstantValues<T>));
  registry->AddSyncKernel(
      "tfrt_dht_sync.set_tensor_with_constant_values." + suffix,
      TFRT_SYNC_KERNEL(SyncSetDenseTensorWithConstantValues<T>));
  registry->AddKernel("tfrt_dht.set_tensor_with_values." + suffix,
                      TFRT_KERNEL(SetDenseTensorWithValues<T>));
  RegisterDhtCreationKernelsForTypeAndRank<T, 0>(registry, t_name);
  RegisterDhtCreationKernelsForTypeAndRank<T, 1>(registry, t_name);
  RegisterDhtCreationKernelsForTypeAndRank<T, 2>(registry, t_name);
  RegisterDhtCreationKernelsForTypeAndRank<T, 3>(registry, t_name);
  RegisterDhtCreationKernelsForTypeAndRank<T, 4>(registry, t_name);
  RegisterDhtCreationKernelsForTypeAndRank<T, 5>(registry, t_name);
}

template <typename T>
static void RegisterDhtComparisonKernelsForType(KernelRegistry* registry,
                                                const std::string& t_name) {
  std::string suffix = t_name;
  registry->AddKernel("tfrt_dht.tensor_equal." + suffix,
                      TFRT_KERNEL(DenseTensorEqual<T>));
  registry->AddKernel("tfrt_dht.tensor_allclose." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T>));
  registry->AddKernel("tfrt_dht.tensor_allclose.3ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 3>));
  registry->AddKernel("tfrt_dht.tensor_allclose.1000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 1000>));
  registry->AddKernel("tfrt_dht.tensor_allclose.2000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 2000>));
  registry->AddKernel("tfrt_dht.tensor_allclose.100000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 100000>));
}

template <typename T>
static void RegisterDenseHostTensorKernelsForType(KernelRegistry* registry,
                                                  const std::string& t_name) {
  RegisterDhtCreationKernelsForType<T>(registry, t_name);
  RegisterDhtComparisonKernelsForType<T>(registry, t_name);
}

void RegisterDenseHostTensorKernels(KernelRegistry* registry) {
  RegisterDenseHostTensorKernelsForType<uint8_t>(registry, "ui8");
  RegisterDenseHostTensorKernelsForType<uint32_t>(registry, "ui32");
  RegisterDenseHostTensorKernelsForType<uint64_t>(registry, "ui64");
  RegisterDenseHostTensorKernelsForType<float>(registry, "f32");
  RegisterDenseHostTensorKernelsForType<double>(registry, "f64");
  RegisterDenseHostTensorKernelsForType<int8_t>(registry, "i8");
  RegisterDenseHostTensorKernelsForType<int32_t>(registry, "i32");
  RegisterDenseHostTensorKernelsForType<int64_t>(registry, "i64");
  RegisterDenseHostTensorKernelsForType<bool>(registry, "bool");
  RegisterDenseHostTensorKernelsForType<std::complex<float>>(registry,
                                                             "complex64");
  RegisterDenseHostTensorKernelsForType<std::complex<double>>(registry,
                                                              "complex128");
  // Only creation kernels for now. Including
  // tfrt/common/compat/eigen/eigen_dtype.h for comparison kernels makes TFRT
  // depend on TF due to b/161569340.
  RegisterDhtCreationKernelsForType<fp16>(registry, "f16");
  RegisterDhtCreationKernelsForType<bf16>(registry, "bf16");

  registry->AddKernel("tfrt_dht.allocate_buffer", TFRT_KERNEL(AllocateBuffer));
  registry->AddSyncKernel("tfrt_dht_sync.allocate_buffer",
                          TFRT_SYNC_KERNEL(SyncAllocateBuffer));
  registry->AddKernel("tfrt_dht.print_tensor", TFRT_KERNEL(PrintTensor));
  registry->AddSyncKernel("tfrt_dht_sync.print_tensor",
                          TFRT_SYNC_KERNEL(SyncPrintTensor));
  registry->AddKernel("tfrt_dht.print_tensor_shape",
                      TFRT_KERNEL(PrintDenseTensorShape));
  registry->AddKernel("tfrt_dht.get_tensor_shape",
                      TFRT_KERNEL(GetDenseTensorShape));
  registry->AddSyncKernel("tfrt_dht_sync.get_tensor_shape",
                          TFRT_SYNC_KERNEL(GetDenseTensorShape));
  registry->AddKernel("tfrt_dht.no_op_ht", TFRT_KERNEL(NoOpHostTensor));
  registry->AddKernel("tfrt_dht.get_buffer", TFRT_KERNEL(GetBuffer));
  registry->AddKernel("tfrt_dht.get_buffer_slice", TFRT_KERNEL(GetBufferSlice));
  registry->AddSyncKernel("tfrt_dht_sync.get_buffer_slice",
                          TFRT_SYNC_KERNEL(GetBufferSlice));
  registry->AddKernel("tfrt_dht.print_buffer", TFRT_KERNEL(PrintBuffer));
}

}  // namespace tfrt
