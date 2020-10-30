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

//===- callback_registry_test.cc --------------------------------*- C++ -*-===//
//
// Unit test for CallbackRegistry.
//
//===----------------------------------------------------------------------===//

#include "tfrt/distributed_runtime/callback_registry.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

std::unique_ptr<std::string> CreateTestData(const size_t size,
                                            const uint8_t initial_value) {
  auto data = std::make_unique<std::string>(size, '\0');
  for (auto i = 0; i < size; ++i) {
    (*data)[i] = i;
  }
  return data;
}

TEST(CallbackRegistry, ValueThenCallback) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key = "key";
  const size_t data_size = 10;
  auto data_ptr1 = CreateTestData(data_size, /*initial_value=*/0);
  auto data_ptr2 = CreateTestData(data_size, /*initial_value=*/data_size);

  auto callback = [&n_calls, data_size](const InstanceKey& key,
                                        std::unique_ptr<std::string> value) {
    n_calls++;
    EXPECT_EQ(value->size(), data_size);
    for (auto i = 0; i < data_size; ++i) {
      EXPECT_EQ(static_cast<uint8_t>((*value)[i]), i);
    }
  };

  rendezvous_registry.SetValue(key, std::move(data_ptr1));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetValue(key, std::move(data_ptr2));
  ASSERT_EQ(n_calls.load(), 1);
}

TEST(CallbackRegistry, CallbackThenValue) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key = "key";
  const size_t data_size = 20;
  auto data_ptr = CreateTestData(data_size, /*initial_value=*/0);

  auto callback = [&n_calls, data_size](const InstanceKey& key,
                                        std::unique_ptr<std::string> value) {
    n_calls++;
    EXPECT_EQ(value->size(), data_size);
    for (auto i = 0; i < data_size; ++i) {
      EXPECT_EQ(static_cast<uint8_t>((*value)[i]), i);
    }
  };

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetValue(key, std::move(data_ptr));
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetCallback(key, callback);
  ASSERT_EQ(n_calls.load(), 1);
}

TEST(CallbackRegistry, MultipleCalls) {
  CallbackRegistry rendezvous_registry{};
  std::atomic<int> n_calls{0};

  InstanceKey key1 = "key1";
  InstanceKey key2 = "key2";
  std::unique_ptr<std::string> data_ptr1;
  std::unique_ptr<std::string> data_ptr2;

  auto callback = [&n_calls](const InstanceKey& key,
                             std::unique_ptr<std::string> data_ptr) {
    n_calls++;
  };

  rendezvous_registry.SetValue(key1, std::move(data_ptr1));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetValue(key2, std::move(data_ptr2));
  ASSERT_EQ(n_calls.load(), 0);

  rendezvous_registry.SetCallback(key1, callback);
  ASSERT_EQ(n_calls.load(), 1);

  rendezvous_registry.SetCallback(key2, callback);
  ASSERT_EQ(n_calls.load(), 2);
}
}  // namespace
}  // namespace tfrt
