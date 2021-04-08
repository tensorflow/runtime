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

// Unit test for TFRT TimerQueue.

#include "tfrt/host_context/timer_queue.h"

#include <chrono>
#include <ctime>
#include <memory>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

using namespace std::chrono_literals;  // NOLINT

// This test checks if timers can expire correctly according to their deadline.
TEST(TimerQueueTest, TimerQueueTimerExpires) {
  tfrt::TimerQueue tq;
  std::atomic<bool> expired_0{false};
  std::atomic<bool> expired_1{false};
  std::atomic<bool> expired_2{false};

  std::atomic<int> global_expiration_index{0};
  std::atomic<int> timer0_expiration_index{0};
  std::atomic<int> timer1_expiration_index{0};

  std::this_thread::sleep_for(1s);

  // Timer1 should expire earlier than Timer0.
  auto timer0 = tq.ScheduleTimer(1s, [&]() {
    expired_0 = true;
    global_expiration_index++;
    timer0_expiration_index = global_expiration_index.load();
  });
  auto timer1 = tq.ScheduleTimer(100ms, [&]() {
    expired_1 = true;
    global_expiration_index++;
    timer1_expiration_index = global_expiration_index.load();
  });
  auto timer2 = tq.ScheduleTimer(1500ms, [&]() { expired_2 = true; });

  std::this_thread::sleep_for(3s);

  // Check if flag and message are set correctly by timer callback.
  ASSERT_TRUE(expired_0);
  ASSERT_TRUE(expired_1);
  ASSERT_TRUE(expired_2);

  // Check if timer1 expires earlier than timer0.
  ASSERT_LT(timer1_expiration_index, timer0_expiration_index);
}

// This test checks if the enqueued timers can be correctly cancelled.
TEST(TimerQueueTest, TimerQueueTimerCancelled) {
  std::atomic<bool> expired_0{false};
  std::atomic<bool> expired_1{false};
  std::atomic<bool> expired_2{false};

  {
    TimerQueue tq;

    // Timer0 and timer1 should be cancelled.
    auto timer0 = tq.ScheduleTimer(800ms, [&]() { expired_0 = true; });
    // Client cancels timer0.
    tq.CancelTimer(timer0);
    auto timer1 = tq.ScheduleTimer(500ms, [&]() { expired_1 = true; });
    auto timer2 = tq.ScheduleTimer(3s, [&]() { expired_2 = true; });

    std::this_thread::sleep_for(2s);

    // TimerQueue goes out of scope here. This should cancel Timer2.
  }

  // Check if timer0 and timer2 are cancelled.
  ASSERT_FALSE(expired_0);
  ASSERT_TRUE(expired_1);
  ASSERT_FALSE(expired_2);
}

}  // namespace
}  // namespace tfrt
