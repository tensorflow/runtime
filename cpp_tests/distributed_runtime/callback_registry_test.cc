/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Unit test for CallbackRegistry.

#include "tfrt/distributed_runtime/callback_registry.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/distributed_runtime/payload.h"

namespace tfrt {
namespace {

Payload CreateTestPayload(const size_t size) {
  llvm::SmallVector<RCReference<HostBuffer>, 4> buffers;
  llvm::SmallVector<uint8_t, 4> data;
  for (auto i = 0; i < size; ++i) {
    data.push_back(i);
  }
  auto data_buffer = data.data();
  auto buffer = tfrt::HostBuffer::CreateFromExternal(
      data_buffer, size, [data = std::move(data)](void*, size_t) {});
  buffers.push_back(std::move(buffer));
  return Payload(std::move(buffers));
}

TEST(CallbackRegistry, ValueThenCallback) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key = "key";
  const size_t data_size = 10;
  auto payload1 = CreateTestPayload(data_size);
  auto payload2 = CreateTestPayload(data_size);

  auto callback = [&n_calls, data_size](const InstanceKey& key, Payload value) {
    n_calls++;
    EXPECT_EQ(value.buffers[0]->size(), data_size);
    for (auto i = 0; i < data_size; ++i) {
      EXPECT_EQ(static_cast<uint8_t*>(value.buffers[0]->data())[i], i);
    }
  };

  rendezvous_registry.SetValue(key, std::move(payload1));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetValue(key, std::move(payload2));
  ASSERT_EQ(n_calls.load(), 1);
}

TEST(CallbackRegistry, CallbackThenValue) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key = "key";
  const size_t data_size = 20;
  auto payload = CreateTestPayload(data_size);

  auto callback = [&n_calls, data_size](const InstanceKey& key, Payload value) {
    n_calls++;
    EXPECT_EQ(value.buffers[0]->size(), data_size);
    for (auto i = 0; i < data_size; ++i) {
      EXPECT_EQ(static_cast<uint8_t*>(value.buffers[0]->data())[i], i);
    }
  };

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetValue(key, std::move(payload));
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 1);
}

TEST(CallbackRegistry, MultipleCalls) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key1 = "key1";
  InstanceKey key2 = "key2";
  Payload payload1 = Payload({});
  Payload payload2 = Payload({});

  auto callback = [&n_calls](const InstanceKey& key, Payload payload) {
    n_calls++;
  };

  rendezvous_registry.SetValue(key1, std::move(payload1));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetValue(key2, std::move(payload2));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetCallback(key1, callback);
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetCallback(key2, callback);
  ASSERT_EQ(n_calls.load(), 2);
}
}  // namespace
}  // namespace tfrt
