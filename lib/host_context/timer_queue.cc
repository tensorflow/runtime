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

// Timer Queue
//
// This file implements TimerQueue.

#include "tfrt/host_context/timer_queue.h"

namespace tfrt {

TimerQueue::TimerQueue() {
  // Start the timer thread.
  // TODO(tfrt-devs): use alternative to std::thread in google-internal build.
  timer_thread_ = std::thread([this]() { TimerThreadRun(); });
}

TimerQueue::~TimerQueue() {
  mu_.lock();
  // Cancel every timer in the queue.
  while (!timers_.empty()) {
    timers_.pop();
  }
  stop_.store(true, std::memory_order_release);
  // Notify the timer thread we are done cleaning up.
  cv_.notify_one();
  mu_.unlock();
  assert(timer_thread_.joinable());
  timer_thread_.join();
}

void TimerQueue::TimerThreadRun() {
  mutex_lock lock(mu_);
  while (!stop_.load(std::memory_order_acquire)) {
    const TimerEntry* top_entry = getTopTimer();
    if (!top_entry) {
      cv_.wait(lock);
    } else {
      // Wait till the timer expires.
      cv_.wait_until(lock, top_entry->deadline_);
    }
    while (!timers_.empty() && timers_.top()->deadline_ <= Clock::now()) {
      const auto top_entry = timers_.top().CopyRef();
      timers_.pop();
      mu_.unlock();
      // If timer is not cancelled, run the callback.
      if (!top_entry->cancelled_) {
        top_entry->timer_callback_();
      }
      mu_.lock();
    }
  }
}

TimerQueue::TimerEntry* TimerQueue::getTopTimer() TFRT_REQUIRES(mu_) {
  while (!timers_.empty()) {
    if (timers_.top()->cancelled_) {
      // Timer is cancelled, discard.
      timers_.pop();
    } else {
      return timers_.top().get();
    }
  }
  // No active timer found. Return a nullptr.
  return nullptr;
}

TimerQueue::TimerHandle TimerQueue::ScheduleTimerAt(TimePoint deadline,
                                                    TimerCallback callback) {
  TimerHandle th = TimerEntry::Create(deadline, std::move(callback));
  bool notify = false;
  {
    mutex_lock lock(mu_);
    // Only notify timer thread when the queue is empty, or when the newly added
    // timer's deadline is shorter.
    notify = timers_.empty() || timers_.top()->deadline_ > deadline;
    timers_.push(th.CopyRef());
  }
  // Notify the timer thread that a new timer is added.
  if (notify) cv_.notify_one();
  return th;
}

TimerQueue::TimerHandle TimerQueue::ScheduleTimer(TimeDuration timeout,
                                                  TimerCallback callback) {
  TimePoint deadline = Clock::now() + timeout;
  return ScheduleTimerAt(deadline, std::move(callback));
}

void TimerQueue::CancelTimer(const TimerQueue::TimerHandle& timer_handle) {
  // TODO(tfrt-dev): make the semantic of CancelTimer() so that if the timer
  // callback has started execution, the CancelTimer() will block until
  // the execution finishes.
  timer_handle->cancelled_.store(true, std::memory_order_release);
}

}  // namespace tfrt
