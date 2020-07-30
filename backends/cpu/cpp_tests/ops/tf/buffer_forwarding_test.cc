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

//===- buffer_forwarding_test.cc --------------------------------*- C++ -*-===//
//
//  Tensorflow operations buffer forwarding unit tests.
//
//===----------------------------------------------------------------------===//

#include "../../../lib/ops/tf/buffer_forwarding.h"

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

static constexpr ResourceContext* kNoResourceCtx = nullptr;

std::unique_ptr<HostContext> CreateTestHostContext() {
  return std::make_unique<HostContext>([](const DecodedDiagnostic&) {},
                                       CreateMallocAllocator(),
                                       CreateSingleThreadedWorkQueue());
}

TEST(BufferForwarding, SameShapeSameType) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();
  ExecutionContext exec_ctx(RequestContext::Create(host, kNoResourceCtx));

  TensorShape shape({1, 2, 3});

  TensorMetadata input_md(GetDType<float>(), shape);
  TensorMetadata output_md(GetDType<float>(), shape);

  auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(input_md, host);
  ASSERT_TRUE(dht.IsUnique());

  Argument<DenseHostTensor> arg(dht.GetAsyncValue());

  {  // Forwards argument to output.
    auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
    ASSERT_EQ(fwd.get().data(), dht.get().data());
  }

  {  // AsyncValue becomes not unique.
    auto copy = dht.CopyRef();
    auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
    ASSERT_NE(fwd.get().data(), dht.get().data());
  }

  {  // DenseHostTensor becomes not unique (not exclusive data owner).
    auto copy = dht.get().CopyRef();
    auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
    ASSERT_NE(fwd.get().data(), dht.get().data());
  }
}

TEST(BufferForwarding, SameShapeDifferentType) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();
  ExecutionContext exec_ctx(RequestContext::Create(host, kNoResourceCtx));

  TensorShape shape({1, 2, 3});

  TensorMetadata input_md(GetDType<float>(), shape);
  TensorMetadata output_md(GetDType<int32_t>(), shape);

  auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(input_md, host);
  ASSERT_TRUE(dht.IsUnique());

  Argument<DenseHostTensor> arg(dht.GetAsyncValue());
  auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
  ASSERT_NE(fwd.get().data(), dht.get().data());
}

TEST(BufferForwarding, DifferentShapeSameType) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();
  ExecutionContext exec_ctx(RequestContext::Create(host, kNoResourceCtx));

  TensorShape shape_in({1, 2, 3});
  TensorShape shape_out({4, 5, 6});

  TensorMetadata input_md(GetDType<float>(), shape_in);
  TensorMetadata output_md(GetDType<float>(), shape_out);

  auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(input_md, host);
  ASSERT_TRUE(dht.IsUnique());

  Argument<DenseHostTensor> arg(dht.GetAsyncValue());
  auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
  ASSERT_NE(fwd.get().data(), dht.get().data());
}

TEST(BufferForwarding, SlicedBuffer) {
  auto host_ptr = CreateTestHostContext();
  HostContext* host = host_ptr.get();
  ExecutionContext exec_ctx(RequestContext::Create(host, kNoResourceCtx));

  TensorShape shape({1, 2, 3});
  TensorMetadata input_md(GetDType<float>(), shape);
  TensorMetadata output_md(GetDType<float>(), shape);

  auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(input_md, host);
  ASSERT_TRUE(dht.IsUnique());

  // Slice a DenseHostTensor of the same type and shape.
  auto sliced_buffer = HostBuffer::CreateFromExternal(dht->buffer().CopyRef(),
                                                      0, dht->buffer()->size());
  auto sliced_dht = host->MakeAvailableAsyncValueRef<DenseHostTensor>(
      input_md, std::move(sliced_buffer));

  Argument<DenseHostTensor> arg(sliced_dht.GetAsyncValue());

  {  // We can't forward slice buffer if `dht` is still alive.
    auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
    ASSERT_NE(fwd.get().data(), dht.get().data());
  }

  {  // When `dht` releases the buffer `sliced_dht` become exclusive data owner.
    void* data_ptr = dht->ReleaseBuffer()->data();
    auto fwd = ForwardInputOrAllocateOutput(exec_ctx, output_md, arg);
    ASSERT_EQ(fwd.get().data(), data_ptr);
  }
}

}  // namespace
}  // namespace tfrt
