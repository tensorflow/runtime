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

//===- event_manager_test.cc ------------------------------------*- C++ -*-===//
//
// Unit test for GPU EventManager
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/event_manager.h"

#include <math.h>

#include <cstring>
#include <ostream>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/latch.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace gpu {
namespace stream {

struct StreamAndBuffers {
  StreamAndBuffers(CurrentContext current, size_t size) : size(size) {
    TFRT_ASSIGN_OR_DIE(stream, StreamCreate(current, StreamFlags::DEFAULT));
    TFRT_ASSIGN_OR_DIE(gpu_buf, MemAlloc(current, size));
    TFRT_ASSIGN_OR_DIE(host_buf_src,
                       MemHostAlloc(current, size, MemHostAllocFlags::DEFAULT));
    TFRT_ASSIGN_OR_DIE(host_buf_dst,
                       MemHostAlloc(current, size, MemHostAllocFlags::DEFAULT));
    std::memset(host_buf_src.get().raw(), 'z', size);
    std::memset(host_buf_dst.get().raw(), 0, size);
  }

  void H2D(CurrentContext current, Event event) {
    EXPECT_TRUE(
        IsSuccess(MemcpyAsync(current, /*dst=*/gpu_buf.get(),
                              /*src=*/host_buf_src.get(), size, stream.get())));
    EXPECT_TRUE(IsSuccess(EventRecord(event, stream.get())));
  }

  void D2H(CurrentContext current, Event event) {
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
  OwningStream stream;
  DeviceMemory<void> gpu_buf;
  HostMemory<void> host_buf_src;
  HostMemory<void> host_buf_dst;
};

class EventManagerTest : public ::testing::Test {
 protected:
  EventManagerTest() : platform_(Platform::CUDA) {}
  void SetUp() override {
    ASSERT_TRUE(IsSuccess(Init(platform_)));
    TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform_));
    ASSERT_GT(count, 0);
    TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform_, 0));
    TFRT_ASSERT_AND_ASSIGN(context_, DevicePrimaryCtxRetain(device));
  }

  // Use 32 MB for copies so that they usually don't complete immediately.
  const size_t kBufferSize = 32 << 20;

  Platform platform_;
  OwningContext context_;
};

TEST_F(EventManagerTest, TestOneStreamOneEvent) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb.H2D(current, event.get());
  sb.D2H(current, event.get());

  latch latch(1);
  manager.Synchronize(event.get(), sb.stream.get(),
                      [&latch, &sb](llvm::Error error) {
                        sb.CheckBuffersEqual();
                        latch.count_down();
                      });
  latch.wait();
}

TEST_F(EventManagerTest, TestOneStreamTwoEvents) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event1,
                         EventCreate(current, EventFlags::DISABLE_TIMING));
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event2,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb.H2D(current, event1.get());
  sb.D2H(current, event2.get());

  latch latch(2);
  manager.Synchronize(event1.get(), sb.stream.get(),
                      [&latch](llvm::Error error) { latch.count_down(); });
  manager.Synchronize(event2.get(), sb.stream.get(),
                      [&latch, &sb](llvm::Error error) {
                        sb.CheckBuffersEqual();
                        latch.count_down();
                      });
  latch.wait();
}

TEST_F(EventManagerTest, TestOneStreamTwoEventsOneByOne) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb(current, kBufferSize);
  {
    TFRT_ASSERT_AND_ASSIGN(OwningEvent event,
                           EventCreate(current, EventFlags::DISABLE_TIMING));

    sb.H2D(current, event.get());
    sb.D2H(current, event.get());

    latch latch(1);
    manager.Synchronize(event.get(), sb.stream.get(),
                        [&latch, &sb](llvm::Error error) {
                          sb.CheckBuffersEqual();
                          latch.count_down();
                        });
    latch.wait();
  }
  {
    sb.ResetHostBuffers();
    TFRT_ASSERT_AND_ASSIGN(OwningEvent event,
                           EventCreate(current, EventFlags::DISABLE_TIMING));

    sb.H2D(current, event.get());
    sb.D2H(current, event.get());

    latch latch(1);
    manager.Synchronize(event.get(), sb.stream.get(),
                        [&latch, &sb](llvm::Error error) {
                          sb.CheckBuffersEqual();
                          latch.count_down();
                        });
    latch.wait();
  }
}

TEST_F(EventManagerTest, TestTwoStreamsTwoEvents) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb1(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event1,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  StreamAndBuffers sb2(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event2,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb1.H2D(current, event1.get());
  sb2.H2D(current, event2.get());
  sb1.D2H(current, event1.get());
  sb2.D2H(current, event2.get());

  latch latch(2);
  manager.Synchronize(event1.get(), sb1.stream.get(),
                      [&latch, &sb1](llvm::Error error) {
                        sb1.CheckBuffersEqual();
                        latch.count_down();
                      });
  manager.Synchronize(event2.get(), sb2.stream.get(),
                      [&latch, &sb2](llvm::Error error) {
                        sb2.CheckBuffersEqual();
                        latch.count_down();
                      });
  latch.wait();
}

TEST_F(EventManagerTest, TestStealing) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb1(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event1,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb1.H2D(current, event1.get());
  sb1.D2H(current, event1.get());

  latch latch1(1);
  manager.Synchronize(event1.get(), sb1.stream.get(),
                      [&latch1, &sb1](llvm::Error error) {
                        sb1.CheckBuffersEqual();
                        latch1.count_down();
                      });
  latch1.wait();

  // stream in sb1 should be idle now. Synchronizing event with stream in
  // sb2 should now steal the thread.

  StreamAndBuffers sb2(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event2,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb2.H2D(current, event2.get());
  sb2.D2H(current, event2.get());

  latch latch2(1);
  manager.Synchronize(event2.get(), sb2.stream.get(),
                      [&latch2, &sb2](llvm::Error error) {
                        sb2.CheckBuffersEqual();
                        latch2.count_down();
                      });
  latch2.wait();
}

TEST_F(EventManagerTest, TestCallSynchronizeFromOnReached) {
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));

  EventManager manager;

  StreamAndBuffers sb1(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event1,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  StreamAndBuffers sb2(current, kBufferSize);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event2,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  sb1.H2D(current, event1.get());
  sb1.D2H(current, event1.get());
  sb2.H2D(current, event2.get());
  sb2.D2H(current, event2.get());

  latch latch(2);
  manager.Synchronize(
      event1.get(), sb1.stream.get(),
      [&latch, &sb1, &manager, &event2, &sb2](llvm::Error error) {
        sb1.CheckBuffersEqual();
        latch.count_down();
        manager.Synchronize(event2.get(), sb2.stream.get(),
                            [&latch, &sb2](llvm::Error error) {
                              sb2.CheckBuffersEqual();
                              latch.count_down();
                            });
      });
  latch.wait();
}

void BM_EventEnqueue(benchmark::State& state) {
  Platform platform(Platform::CUDA);
  ASSERT_TRUE(IsSuccess(Init(platform)));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  EventManager manager;

  StreamAndBuffers sb(current, /*size=*/4);
  TFRT_ASSERT_AND_ASSIGN(OwningEvent event,
                         EventCreate(current, EventFlags::DISABLE_TIMING));

  {
    latch latch(1);
    manager.Synchronize(event.get(), sb.stream.get(),
                        [&latch](llvm::Error error) { latch.count_down(); });
    latch.wait();
  }
  for (auto _ : state) {
    manager.Synchronize(event.get(), sb.stream.get(), [](llvm::Error error) {
      TFRT_LOG_IF(FATAL, error.success())
          << "Hit an error in calback: " << error;
    });
  }
  {
    latch latch(1);
    manager.Synchronize(event.get(), sb.stream.get(),
                        [&latch](llvm::Error error) { latch.count_down(); });
    latch.wait();
  }
}
BENCHMARK(BM_EventEnqueue);

}  // namespace stream
}  // namespace gpu
}  // namespace tfrt
