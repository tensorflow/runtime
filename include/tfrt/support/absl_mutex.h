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

// Wrappers around std::{mutex,unique_lock,condition_variable} with support for
// thread safety annotations.

#ifndef TFRT_SUPPORT_ABSL_MUTEX_H_
#define TFRT_SUPPORT_ABSL_MUTEX_H_

#include <chrono>

#include "tfrt/support/thread_annotations.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/time/time.h"

// Avoid conflict with @org_tensorflow/core/platform/mutex.h:181
// TODO(tfrt-devs): remove the macro in @org_tensorflow/core/platform/mutex.h
// and replace it with the [nodiscard] attribute when c++17 is allowed.
#undef mutex_lock

namespace tfrt {

// Wrap absl::mutex with support for thread annotations.
class TFRT_CAPABILITY("mutex") mutex {
 public:
  mutex() = default;
  ~mutex() = default;

  mutex(const mutex&) = delete;
  mutex& operator=(const mutex&) = delete;

  void lock() TFRT_ACQUIRE() { mu_.Lock(); }
  void unlock() TFRT_RELEASE() { mu_.Unlock(); }

 private:
  friend class mutex_lock;
  absl::Mutex mu_;
};

class TFRT_SCOPED_CAPABILITY mutex_lock {
 public:
  explicit mutex_lock(mutex& mu) TFRT_ACQUIRE(mu) : mu_(&mu.mu_) {
    mu_->Lock();
  }
  ~mutex_lock() TFRT_RELEASE() { mu_->Unlock(); }

  mutex_lock(const mutex_lock&) = delete;
  mutex_lock& operator=(const mutex_lock&) = delete;

 private:
  friend class condition_variable;
  absl::Mutex* const mu_;
};

// Wraps absl::CondVar with support for mutex_lock.
class condition_variable {
 public:
  condition_variable() = default;
  ~condition_variable() = default;

  condition_variable(const condition_variable&) = delete;
  condition_variable& operator=(const condition_variable&) = delete;

  void wait(mutex_lock& mu) TFRT_NO_THREAD_SAFETY_ANALYSIS { cv_.Wait(mu.mu_); }

  template <class Predicate>
  void wait(mutex_lock& mu, Predicate pred) TFRT_NO_THREAD_SAFETY_ANALYSIS {
    while (!pred()) wait(mu);
  }

  template <class Clock, class Duration, class Predicate>
  bool wait_until(mutex_lock& mu,
                  const std::chrono::time_point<Clock, Duration>& timeout_time,
                  Predicate pred) TFRT_NO_THREAD_SAFETY_ANALYSIS {
    const auto timeout_time_converted =
        std::chrono::time_point_cast<std::chrono::microseconds>(timeout_time);
    absl::Time deadline = absl::FromChrono(timeout_time_converted);

    while (!pred()) {
      // CondVar timed out on wait
      if (cv_.WaitWithDeadline(mu.mu_, deadline)) {
        return pred();
      }

      // Need to check time out here as CondVar::WaitWithDeadline may return
      // true or false if the deadline has already passed.
      if (absl::Now() > deadline) return pred();
    }
    return true;
  }

  template <class Clock, class Duration>
  bool wait_until(mutex_lock& mu,
                  const std::chrono::time_point<Clock, Duration>& timeout_time)
      TFRT_NO_THREAD_SAFETY_ANALYSIS {
    auto timeout_time_converted =
        std::chrono::time_point_cast<std::chrono::microseconds>(timeout_time);
    absl::Time deadline = absl::FromChrono(timeout_time_converted);
    return cv_.WaitWithDeadline(mu.mu_, deadline);
  }

  void notify_one() { cv_.Signal(); }
  void notify_all() { cv_.SignalAll(); }

 private:
  absl::CondVar cv_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_ABSL_MUTEX_H_
