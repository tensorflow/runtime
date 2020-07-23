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

//===- cwise_binary_kernel_test.cc ------------------------------*- C++ -*-===//
//
//  Column wise binary kernels tests and benchmarks.
//
//===----------------------------------------------------------------------===//

#include "../../lib/kernels/cwise_binary_kernels.h"

#include "benchmark/benchmark.h"
#include "tfrt/common/ops/tf/bcast.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/latch.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {
std::unique_ptr<HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(num_threads, num_threads));
}
}  // namespace

void BinaryKernel(benchmark::State& state, int num_threads,
                  const TensorShape& lhs_shape, const TensorShape& rhs_shape) {
  auto host = CreateTestHostContext(num_threads);
  ExecutionContext exec_ctx(
      RequestContext::Create(host.get(), /*resource_context=*/nullptr));

  TensorShape res_shape = GetBroadcastedShape(lhs_shape, rhs_shape).get();

  TensorMetadata lhs_md(GetDType<float>(), TensorShape(lhs_shape));
  TensorMetadata rhs_md(GetDType<float>(), TensorShape(rhs_shape));
  TensorMetadata res_md(GetDType<float>(), TensorShape(res_shape));

  auto lhs = DenseHostTensor::CreateUninitialized(lhs_md, host.get());
  auto rhs = DenseHostTensor::CreateUninitialized(rhs_md, host.get());
  auto res = DenseHostTensor::CreateUninitialized(res_md, host.get());

  using Functor = typename ::tfrt::cpu::functor::Add::Functor<float>;

  for (auto _ : state) {
    tfrt::latch done(1);

    ::tfrt::cpu::BinaryKernel<Functor>(*lhs, *rhs, res.getPointer(), exec_ctx,
                                       [&](Error err) { done.count_down(); });

    done.wait();
  }

  state.SetItemsProcessed(res_shape.GetNumElements() * state.iterations());
}

void AddTensorScalar(benchmark::State& state, int num_threads,
                     ArrayRef<ssize_t> tensor_dims) {
  TensorShape lhs_shape(tensor_dims);
  TensorShape rhs_shape({});
  BinaryKernel(state, num_threads, lhs_shape, rhs_shape);
}

#define BM_Add_TensorD2_Scalar(threads, D0, D1)                  \
  static void BM_AddTensor_##D0##x##D1##_Scalar_tpool_##threads( \
      benchmark::State& state) {                                 \
    AddTensorScalar(state, threads, {D0, D1});                   \
  }                                                              \
  BENCHMARK(BM_AddTensor_##D0##x##D1##_Scalar_tpool_##threads)

// [1, 1] + []
BM_Add_TensorD2_Scalar(4, 1, 1);
BM_Add_TensorD2_Scalar(8, 1, 1);
BM_Add_TensorD2_Scalar(16, 1, 1);

// [300, 1] + []
BM_Add_TensorD2_Scalar(4, 300, 1);
BM_Add_TensorD2_Scalar(8, 300, 1);
BM_Add_TensorD2_Scalar(16, 300, 1);

// [300, 300] + []
BM_Add_TensorD2_Scalar(4, 300, 300);
BM_Add_TensorD2_Scalar(8, 300, 300);
BM_Add_TensorD2_Scalar(16, 300, 300);

// [1500, 300] + []
BM_Add_TensorD2_Scalar(4, 1500, 300);
BM_Add_TensorD2_Scalar(8, 1500, 300);
BM_Add_TensorD2_Scalar(16, 1500, 300);

}  // namespace tfrt
