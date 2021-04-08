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

// Tests for tfrt::latch implementation.

#include "tfrt/support/latch.h"

#include <thread>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(LatchTest, ConstructedReady) {
  latch latch(0);
  ASSERT_TRUE(latch.try_wait());
  latch.wait();
}

TEST(LatchTest, CountDownFromOne) {
  latch latch(1);
  ASSERT_FALSE(latch.try_wait());
  latch.count_down(1);
  ASSERT_TRUE(latch.try_wait());
  latch.wait();
}

TEST(LatchTest, CountDownFromThree) {
  latch latch(3);
  ASSERT_FALSE(latch.try_wait());
  latch.count_down(1);
  ASSERT_FALSE(latch.try_wait());
  latch.count_down(2);
  ASSERT_TRUE(latch.try_wait());
  latch.wait();
}

TEST(LatchTest, ConcurrentCountDown) {
  latch latch(1000);

  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 100; ++j) latch.count_down();
    });
  }

  latch.wait();

  for (auto& thread : threads) thread.join();
}

}  // namespace
}  // namespace tfrt
