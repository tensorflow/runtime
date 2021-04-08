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

// Unit test for GPU EventManager
#include "tfrt/gpu/event_manager.h"

#include <math.h>

#include <cstring>
#include <ostream>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace gpu {

struct StreamAndBuffers {
  StreamAndBuffers(stream::CurrentContext current, size_t size) : size(size) {
    TFRT_ASSIGN_OR_DIE(stream,
                       StreamCreate(current, stream::StreamFlags::DEFAULT));
    TFRT_ASSIGN_OR_DIE(gpu_buf, MemAlloc(current, size));
    TFRT_ASSIGN_OR_DIE(
        host_buf_src,
        MemHostAlloc(current, size, stream::MemHostAllocFlags::DEFAULT));
    TFRT_ASSIGN_OR_DIE(
        host_buf_dst,
        MemHostAlloc(current, size, stream::MemHostAllocFlags::DEFAULT));
    std::memset(host_buf_src.get().raw(), 'z', size);
    std::memset(host_buf_dst.get().raw(), 0, size);
  }

  void H2D(stream::CurrentContext current, stream::Event event) {
    EXPECT_TRUE(
        IsSuccess(MemcpyAsync(current, /*dst=*/gpu_buf.get(),
                              /*src=*/host_buf_src.get(), size, stream.get())));
    EXPECT_TRUE(IsSuccess(EventRecord(event, stream.get())));
  }

  void D2H(stream::CurrentContext current, stream::Event event) {
    EXPECT_TRUE(
        IsSuccess(MemcpyAsync(current, /*dst=*/host_buf_dst.get(),
                              /*src=*/gpu_buf.get(), size, stream.get())));
    EXPECT_TRUE(IsSuccess(EventRecord(event, stream.get())));
  }

  void CheckBuffersEqual() {
    EXPECT_EQ(0, std::memcmp(host_buf_dst.get().raw(), host_buf_src.get().raw(),
                             size));
  }

  void ResetHostBuffers() {
    std::memset(host_buf_src.get().raw(), 'z', size);
    std::memset(host_buf_dst.get().raw(), 0, size);
  }

  size_t size;
  stream::OwningStream stream;
  stream::DeviceMemory<void> gpu_buf;
  stream::HostMemory<void> host_buf_src;
  stream::HostMemory<void> host_buf_dst;
};

static RCReference<RcEvent> CreateRcEvent(stream::CurrentContext current) {
  stream::OwningEvent event =
      std::move(*EventCreate(current, stream::EventFlags::DISABLE_TIMING));
  return TakeRef(new RcEvent(std::move(event)));
}

class EventManagerTest : public ::testing::Test {
 protected:
  EventManagerTest() : platform_(stream::Platform::CUDA) {}
  void SetUp() override {
    ASSERT_TRUE(IsSuccess(Init(platform_)));
    TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform_));
    ASSERT_GT(count, 0);
    TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform_, 0));
    TFRT_ASSERT_AND_ASSIGN(context_, DevicePrimaryCtxRetain(device));
  }

  // Use 32 MB for copies so that they usually don't complete immediately.
  const size_t kBufferSize = 32 << 20;

  stream::Platform platform_;
  stream::OwningContext context_;
};

static std::unique_ptr<HostContext> CreateHostContext() {
  return std::make_unique<HostContext>([](const DecodedDiagnostic&) {},
                                       CreateMallocAllocator(),
                                       CreateSingleThreadedWorkQueue());
}

static std::unique_ptr<HostContext> CreateHostContext(int thread_count) {
  return std::make_unique<HostContext>(
      [](const DecodedDiagnostic&) {}, CreateMallocAllocator(),
      CreateMultiThreadedWorkQueue(thread_count, thread_count));
}

TEST_F(EventManagerTest, TestOneStreamOneEvent) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));
  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb(current, kBufferSize);
  auto event = CreateRcEvent(current);

  sb.H2D(current, event->get());
  sb.D2H(current, event->get());

  auto async_value = manager.Synchronize(std::move(event));
  host_context->Await(async_value.CopyRCRef());
  sb.CheckBuffersEqual();
}

TEST_F(EventManagerTest, TestOneStreamTwoEvents) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb(current, kBufferSize);
  auto event1 = CreateRcEvent(current);
  auto event2 = CreateRcEvent(current);

  sb.H2D(current, event1->get());
  sb.D2H(current, event2->get());

  auto async_value1 = manager.Synchronize(event1.CopyRef());
  auto async_value2 = manager.Synchronize(event2.CopyRef());

  host_context->Await({async_value1.CopyRCRef(), async_value2.CopyRCRef()});
  sb.CheckBuffersEqual();
}

TEST_F(EventManagerTest, TestOneStreamTwoEventsOneByOne) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb(current, kBufferSize);
  {
    auto event = CreateRcEvent(current);

    sb.H2D(current, event->get());
    sb.D2H(current, event->get());

    auto async_value = manager.Synchronize(event.CopyRef());
    host_context->Await(async_value.CopyRCRef());
    sb.CheckBuffersEqual();
  }
  {
    sb.ResetHostBuffers();
    auto event = CreateRcEvent(current);

    sb.H2D(current, event->get());
    sb.D2H(current, event->get());

    auto async_value = manager.Synchronize(event.CopyRef());
    host_context->Await(async_value.CopyRCRef());
    sb.CheckBuffersEqual();
  }
}

TEST_F(EventManagerTest, TestTwoStreamsTwoEvents) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb1(current, kBufferSize);
  auto event1 = CreateRcEvent(current);

  StreamAndBuffers sb2(current, kBufferSize);
  auto event2 = CreateRcEvent(current);

  sb1.H2D(current, event1->get());
  sb2.H2D(current, event2->get());
  sb1.D2H(current, event1->get());
  sb2.D2H(current, event2->get());

  auto async_value1 = manager.Synchronize(event1.CopyRef());
  auto async_value2 = manager.Synchronize(event2.CopyRef());
  host_context->Await({async_value1.CopyRCRef(), async_value2.CopyRCRef()});
  sb1.CheckBuffersEqual();
  sb2.CheckBuffersEqual();
}

TEST_F(EventManagerTest, TestCallSynchronizeFromAndThen) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb1(current, kBufferSize);
  auto event1 = CreateRcEvent(current);

  StreamAndBuffers sb2(current, kBufferSize);
  auto event2 = CreateRcEvent(current);

  sb1.H2D(current, event1->get());
  sb1.D2H(current, event1->get());
  sb2.H2D(current, event2->get());
  sb2.D2H(current, event2->get());

  auto async_value1 = manager.Synchronize(event1.CopyRef());
  auto async_value2 = MakeUnconstructedAsyncValueRef<Chain>(host_context.get());
  async_value1.AndThen([&] {
    sb1.CheckBuffersEqual();
    manager.Synchronize(event2.CopyRef()).AndThen([&] {
      sb2.CheckBuffersEqual();
      async_value2.emplace();
    });
  });
  host_context->Await({async_value1.CopyRCRef(), async_value2.CopyRCRef()});
  host_context->Quiesce();
}

void BM_EventEnqueue(benchmark::State& state) {
  stream::Platform platform(stream::Platform::CUDA);
  ASSERT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  auto host_context = CreateHostContext();
  EventManager manager(*host_context);

  StreamAndBuffers sb(current, /*size=*/4);
  auto event = CreateRcEvent(current);

  ASSERT_TRUE(IsSuccess(stream::EventRecord(event->get(), sb.stream.get())));

  host_context->Await(manager.Synchronize(event.CopyRef()).CopyRCRef());
  for (auto _ : state) {
    manager.Synchronize(event.CopyRef());
  }
  host_context->Await(manager.Synchronize(event.CopyRef()).CopyRCRef());
}

void BM_ManyThreadsManyStreams(benchmark::State& state) {
  stream::Platform platform(stream::Platform::CUDA);
  ASSERT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  const int thread_count = state.range(0);
  const size_t size = state.range(1);
  auto host_context = CreateHostContext(thread_count);
  EventManager manager(*host_context);

  std::vector<StreamAndBuffers> sb;
  sb.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    sb.emplace_back(current, size);
  }
  std::vector<RCReference<RcEvent>> events(thread_count);
  std::generate(events.begin(), events.end(),
                [&] { return CreateRcEvent(current); });

  std::vector<RCReference<AsyncValue>> chains(thread_count);
  for (auto _ : state) {
    for (int i = 0; i < thread_count; ++i) {
      sb[i].H2D(current, events[i]->get());
    }
    std::transform(events.begin(), events.end(), chains.begin(),
                   [&](const auto& event) {
                     return manager.Synchronize(event.CopyRef());
                   });
    host_context->Await(chains);
  }
}

void BM_MoreStreamsThanThreads(benchmark::State& state) {
  stream::Platform platform(stream::Platform::CUDA);
  ASSERT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  const int thread_count = state.range(0);
  const int stream_count = state.range(1);
  const size_t size = 1 << 20;
  auto host_context = CreateHostContext(thread_count);
  EventManager manager(*host_context);

  std::vector<StreamAndBuffers> sb;
  sb.reserve(stream_count);
  for (int i = 0; i < stream_count; ++i) {
    sb.emplace_back(current, size);
  }
  std::vector<RCReference<RcEvent>> events(stream_count);
  std::generate(events.begin(), events.end(),
                [&] { return CreateRcEvent(current); });

  std::vector<RCReference<AsyncValue>> chains(stream_count);
  for (auto _ : state) {
    for (int i = 0; i < stream_count; ++i) {
      sb[i].H2D(current, events[i]->get());
    }
    std::transform(events.begin(), events.end(), chains.begin(),
                   [&](const auto& event) {
                     return manager.Synchronize(std::move(event.CopyRef()));
                   });
    host_context->Await(chains);
  }
}

void BM_MultipleEventsPerStream(benchmark::State& state) {
  stream::Platform platform(stream::Platform::CUDA);
  ASSERT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  const int stream_count = state.range(0);
  const int events_per_stream = state.range(1);
  auto host_context = CreateHostContext(stream_count);
  EventManager manager(*host_context);

  std::vector<StreamAndBuffers> sb;
  sb.reserve(stream_count);
  for (int i = 0; i < stream_count; ++i) {
    sb.emplace_back(current, /*size=*/8192);
  }

  std::vector<RCReference<RcEvent>> events(stream_count * events_per_stream);
  std::generate(events.begin(), events.end(),
                [&] { return CreateRcEvent(current); });

  std::vector<RCReference<AsyncValue>> chains(events.size());
  for (auto _ : state) {
    for (int j = 0; j < events_per_stream; ++j) {
      for (int i = 0; i < stream_count; ++i) {
        sb[i].H2D(current, events[i * events_per_stream + j]->get());
      }
    }
    std::transform(events.begin(), events.end(), chains.begin(),
                   [&](const auto& event) {
                     return manager.Synchronize(std::move(event.CopyRef()));
                   });

    host_context->Await(chains);
  }
}

BENCHMARK(BM_EventEnqueue);
BENCHMARK(BM_ManyThreadsManyStreams)->RangePair(1, 32, 4, 32 << 20);
BENCHMARK(BM_MultipleEventsPerStream)->RangePair(1, 32, 2, 32);
BENCHMARK(BM_MoreStreamsThanThreads)->RangePair(1, 32, 2, 10);

}  // namespace gpu
}  // namespace tfrt
