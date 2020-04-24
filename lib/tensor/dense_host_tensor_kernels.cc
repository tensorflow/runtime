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

//===- dense_host_tensor_kernels.cc -----------------------------*- C++ -*-===//
//
// This file defines the kernels for dense host tensors.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/dense_host_tensor_kernels.h"

#include <complex>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ostream.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

template <typename T, size_t Rank>
static void CreateUninitializedDenseTensor(Result<DenseHostTensor> out,
                                           FlatArrayAttribute<ssize_t> shape_in,
                                           KernelErrorHandler handler,
                                           KernelFrame* frame) {
  auto result = DenseHostTensor::CreateUninitialized<T>(
      TensorShape(shape_in.data()), frame->GetHostContext());
  if (!result.hasValue()) {
    handler.ReportError("Cannot allocate tensor");
    return;
  }
  out.Emplace(std::move(*result));
}

template <typename T>
static void MakeTensor(Argument<RCReference<HostBuffer>> buffer,
                       Argument<TensorShape> shape, Argument<Chain> in_chain,
                       Result<DenseHostTensor> tensor, Result<Chain> out_chain,
                       KernelErrorHandler handler) {
  if (buffer.get()->size() !=
      shape->GetNumElements() * GetDType<T>().GetHostSize()) {
    std::string error_msg;
    llvm::raw_string_ostream ss(error_msg);
    ss << "dht.make_tensor failed: buffer_size (" << buffer.get()->size()
       << ") is not equal to the number of elements in shape (" << *shape
       << ") times element size (" << GetDType<T>().GetHostSize() << ")";
    handler.ReportError(ss.str());
    return;
  }
  tensor.Emplace(TensorMetadata(GetDType<T>(), *shape), std::move(*buffer));
  out_chain.Set(in_chain);
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
    Result<Chain> chain_out, FlatArrayAttribute<T> values,
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

static Chain PrintTensor(const Tensor& t) {
  tfrt::outs() << t << "\n";
  tfrt::outs().flush();
  return Chain();
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
static void DenseTensorEqual(ArgumentView<MutableDHTArrayView<T>> t1,
                             ArgumentView<MutableDHTArrayView<T>> t2,
                             Argument<Chain> chain, Result<bool> output1,
                             Result<Chain> output2) {
  output1.Emplace(*t1 == *t2);
  // Reuse input chain.
  output2.Set(chain);
}

// Returns true if two input tensors are near enough to each other.
template <typename T, int ULP = 2>
static void DenseTensorAllClose(ArgumentView<MutableDHTArrayView<T>> t1,
                                ArgumentView<MutableDHTArrayView<T>> t2,
                                Argument<Chain> chain, Result<bool> output1,
                                Result<Chain> output2) {
  output1.Emplace(AllElementsClose<T, ULP>(*t1, *t2));
  // Reuse input chain.
  output2.Set(chain);
}

static void GetBuffer(Argument<DenseHostTensor> t, Argument<Chain> chain_in,
                      Result<RCReference<HostBuffer>> buffer,
                      Result<Chain> chain_out) {
  buffer.Emplace(t->buffer().CopyRef());
  chain_out.Set(chain_in);
}

static void PrintBuffer(Argument<RCReference<HostBuffer>> buffer,
                        Argument<Chain> chain_in, Result<Chain> chain_out) {
  tfrt::outs() << **buffer << "\n";
  tfrt::outs().flush();
  chain_out.Set(chain_in);
}

template <typename T, size_t Rank>
static void SliceInPlaceHelper(DHTIndexableView<T, Rank> input,
                               const int64_t* begin,
                               MutableDHTIndexableView<T, Rank> output,
                               std::array<size_t, Rank>& input_coord,
                               std::array<size_t, Rank>& output_coord,
                               int cur_coord_idx) {
  const FixedRankShape<Rank>& output_shape = output.FixedShape();
  if (cur_coord_idx == Rank) {
    output[output_coord] = input[input_coord];
    return;
  }

  for (int64_t i = 0, e = output_shape[cur_coord_idx]; i < e; i++) {
    input_coord[cur_coord_idx] = begin[cur_coord_idx] + i;
    output_coord[cur_coord_idx] = i;
    SliceInPlaceHelper<T, Rank>(input, begin, output, input_coord, output_coord,
                                cur_coord_idx + 1);
  }
}

// TODO(donglin): optimize this kernel by using Eigen library APIs.
// Sets output = tf.slice(input, begin, output.shape()) where both input and
// output are tensors with 'Rank' dimensions. 'begin' should be a vector of size
// 'Rank'.
template <typename T, size_t Rank>
static llvm::Expected<Chain> SliceInPlace(
    DHTIndexableView<T, Rank> input, DHTIndexableView<int64_t, 1> begin,
    const Chain& chain_in, MutableDHTIndexableView<T, Rank> output) {
  if (begin.NumElements() != Rank) {
    return MakeStringError(
        "the size of 'begin_index' should match the input tensor rank");
  }
  const int64_t* begin_data = static_cast<const int64_t*>(begin.data());
  std::array<size_t, Rank> input_coord, output_coord;
  SliceInPlaceHelper<T, Rank>(input, begin_data, output, input_coord,
                              output_coord, 0);

  return Chain();
}

template <typename T, size_t Rank>
static void RegisterDenseHostTensorKernelsForTypeAndRank(
    KernelRegistry* registry, const std::string& t_name) {
  std::string suffix = t_name + "." + std::to_string(Rank);
  registry->AddKernel("dht.create_uninitialized_tensor." + suffix,
                      TFRT_KERNEL(CreateUninitializedDenseTensor<T, Rank>));
}

template <typename T>
static void RegisterDenseHostTensorKernelsForType(KernelRegistry* registry,
                                                  const std::string& t_name) {
  std::string suffix = t_name;
  // Constant is in the name because we will presumably want a version that uses
  // a variable.
  registry->AddKernel("dht.fill_tensor_with_constant." + suffix,
                      TFRT_KERNEL(FillDenseTensorWithConstantValue<T>));
  registry->AddKernel("dht.make_tensor." + suffix, TFRT_KERNEL(MakeTensor<T>));
  registry->AddKernel("dht.set_tensor_with_constant_values." + suffix,
                      TFRT_KERNEL(SetDenseTensorWithConstantValues<T>));
  registry->AddKernel("dht.set_tensor_with_values." + suffix,
                      TFRT_KERNEL(SetDenseTensorWithValues<T>));
  registry->AddKernel("dht.tensor_equal." + suffix,
                      TFRT_KERNEL(DenseTensorEqual<T>));
  registry->AddKernel("dht.tensor_allclose." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T>));
  registry->AddKernel("dht.tensor_allclose.3ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 3>));
  registry->AddKernel("dht.tensor_allclose.1000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 1000>));
  registry->AddKernel("dht.tensor_allclose.2000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 2000>));
  registry->AddKernel("dht.tensor_allclose.100000ulp." + suffix,
                      TFRT_KERNEL(DenseTensorAllClose<T, 100000>));
  registry->AddKernel("dht.slice_inplace." + suffix + ".3",
                      TFRT_KERNEL(SliceInPlace<T, 3>));
  RegisterDenseHostTensorKernelsForTypeAndRank<T, 0>(registry, t_name);
  RegisterDenseHostTensorKernelsForTypeAndRank<T, 1>(registry, t_name);
  RegisterDenseHostTensorKernelsForTypeAndRank<T, 2>(registry, t_name);
  RegisterDenseHostTensorKernelsForTypeAndRank<T, 3>(registry, t_name);
  RegisterDenseHostTensorKernelsForTypeAndRank<T, 4>(registry, t_name);
}

void RegisterDenseHostTensorKernels(KernelRegistry* registry) {
  RegisterDenseHostTensorKernelsForType<uint8_t>(registry, "ui8");
  RegisterDenseHostTensorKernelsForType<float>(registry, "f32");
  RegisterDenseHostTensorKernelsForType<int32_t>(registry, "i32");
  RegisterDenseHostTensorKernelsForType<int64_t>(registry, "i64");
  RegisterDenseHostTensorKernelsForType<bool>(registry, "bool");
  RegisterDenseHostTensorKernelsForType<std::complex<float>>(registry,
                                                             "complex64");
  registry->AddKernel("dht.print_tensor", TFRT_KERNEL(PrintTensor));
  registry->AddKernel("dht.print_tensor_shape",
                      TFRT_KERNEL(PrintDenseTensorShape));
  registry->AddKernel("dht.get_tensor_shape", TFRT_KERNEL(GetDenseTensorShape));
  registry->AddKernel("dht.no_op_ht", TFRT_KERNEL(NoOpHostTensor));
  registry->AddKernel("dht.get_buffer", TFRT_KERNEL(GetBuffer));
  registry->AddKernel("dht.print_buffer", TFRT_KERNEL(PrintBuffer));
}

}  // namespace tfrt
