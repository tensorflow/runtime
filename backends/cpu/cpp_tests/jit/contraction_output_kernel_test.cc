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

//===- contraction_output_kernel_test.cc ------------------------*- C++ -*-===//
//
//  Jit Contraction Output Kernel tests.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/contraction_output_kernel.h"

#include <sys/types.h>

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace {

std::unique_ptr<HostContext> CreateTestHostContext() {
  return std::make_unique<HostContext>([](const DecodedDiagnostic&) {},
                                       CreateMallocAllocator(),
                                       CreateSingleThreadedWorkQueue());
}

// We don't use TFRT EigenTensor because for the contraction output kernel
// we need column major data layout.
template <typename T, size_t Rank = 1>
using EigenTensor =
    Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::ColMajor, Eigen::Index>,
                     Eigen::Aligned>;

TEST(ContractionOutputKernelTest, AddOne) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();

  auto f32 = DType(DType::F32);

  std::vector<float> storage(100);
  EigenTensor<float, 2> tensor(storage.data(), 10, 10);

  OpAttrs attrs;
  OpAttrsRef attrs_ref = attrs.freeze();

  // Column major mapper for the `tensor`.
  using ContractionOutputMapper =
      cpu::jit::ContractionOutputKernel<float>::ContractionOutputMapper;
  ContractionOutputMapper mapper(storage.data(), 10, 1);

  // Compile contraction output kernel.
  auto kernel = cpu::jit::GetCompiledContractionOutputKernel(
      host, {"AddOne"}, attrs_ref, f32, /*additional_args=*/{});
  ASSERT_FALSE(static_cast<bool>(kernel.takeError()));

  // Call compiled contraction output kernel.
  cpu::jit::ContractionOutputKernel<float> output_kernel(*kernel, {});
  output_kernel(mapper, {true}, 0, 0, 10, 10);

  // All values must be increased by one.
  ASSERT_TRUE(std::all_of(storage.begin(), storage.end(),
                          [](float value) { return value == 1.0; }));

  // Increase by one a slice of the tensor.
  output_kernel(mapper, {true}, 0, 0, 5, 3);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_EQ(tensor(i, j), 2);
    }
  }
}

TEST(ContractionOutputKernelTest, AddBias) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();

  auto f32 = DType(DType::F32);

  std::vector<float> storage(100);
  EigenTensor<float, 2> tensor(storage.data(), 10, 10);

  const float bias_initial_value = 11.0;
  std::vector<float> bias_storage(10);
  std::iota(bias_storage.begin(), bias_storage.end(), bias_initial_value);

  auto noop_deallocator = [](void* ptr, size_t size) {};
  TensorShape bias_shape(ArrayRef<ssize_t>(10));
  DenseHostTensor bias(TensorMetadata(f32, bias_shape),
                       HostBuffer::CreateFromExternal(bias_storage.data(), 10,
                                                      noop_deallocator));

  OpAttrs attrs;
  OpAttrsRef attrs_ref = attrs.freeze();

  // Column major mapper for the `tensor`.
  using ContractionOutputMapper =
      cpu::jit::ContractionOutputKernel<float>::ContractionOutputMapper;

  const int row_offset = 2;
  const int col_offset = 3;

  float* mapper_base = storage.data() + 10 * col_offset + row_offset;
  ContractionOutputMapper mapper(mapper_base, 10, 1);

  // Compile contraction output kernel.
  auto kernel = cpu::jit::GetCompiledContractionOutputKernel(
      host, {"BiasAdd"}, attrs_ref, f32, /*additional_args=*/{f32});
  ASSERT_FALSE(static_cast<bool>(kernel.takeError()));

  // Call compiled contraction output kernel.
  cpu::jit::ContractionOutputKernel<float> output_kernel(*kernel, {&bias});
  output_kernel(mapper, {true}, row_offset, col_offset, 5, 3);

  for (int i = 3; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_EQ(tensor(row_offset + i, col_offset + j),
                bias_initial_value + row_offset + i);
    }
  }
}

TEST(ContractionOutputKernelTest, AddOneAndBias) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();

  auto f32 = DType(DType::F32);

  std::vector<float> storage(100);
  EigenTensor<float, 2> tensor(storage.data(), 10, 10);

  const float bias_initial_value = 11.0;
  std::vector<float> bias_storage(10);
  std::iota(bias_storage.begin(), bias_storage.end(), bias_initial_value);

  auto noop_deallocator = [](void* ptr, size_t size) {};
  TensorShape bias_shape(ArrayRef<ssize_t>(10));
  DenseHostTensor bias(TensorMetadata(f32, bias_shape),
                       HostBuffer::CreateFromExternal(bias_storage.data(), 10,
                                                      noop_deallocator));

  OpAttrs attrs;
  OpAttrsRef attrs_ref = attrs.freeze();

  // Column major mapper for the `tensor`.
  using ContractionOutputMapper =
      cpu::jit::ContractionOutputKernel<float>::ContractionOutputMapper;

  const int row_offset = 2;
  const int col_offset = 3;

  float* mapper_base = storage.data() + 10 * col_offset + row_offset;
  ContractionOutputMapper mapper(mapper_base, 10, 1);

  // Compile contraction output kernel.
  auto kernel = cpu::jit::GetCompiledContractionOutputKernel(
      host, {"AddOne", "BiasAdd"}, attrs_ref, f32, /*additional_args=*/{f32});
  ASSERT_FALSE(static_cast<bool>(kernel.takeError()));

  // Call compiled contraction output kernel.
  cpu::jit::ContractionOutputKernel<float> output_kernel(*kernel, {&bias});
  output_kernel(mapper, {true}, row_offset, col_offset, 5, 3);

  for (int i = 3; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_EQ(tensor(row_offset + i, col_offset + j),
                1.0 + bias_initial_value + row_offset + i);
    }
  }
}

}  // namespace
}  // namespace tfrt
