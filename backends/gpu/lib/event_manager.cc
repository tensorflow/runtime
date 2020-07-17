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

//===- event_manager.cc - Utility for CUDA/HIP event waiting  ---*- C++ -*-===//
//
// This file implements the EventManager class that can be used to get
// notification on the host when some GPU events have been reached.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/event_manager.h"

#include <atomic>
#include <memory>

#include "tfrt/support/logging.h"

namespace tfrt {
namespace gpu {

EventManager::EventManager() {}

void EventManager::ThreadInfo::AddEvent(stream::Event event,
                                        EventReachedCallback on_reached) {
  bool was_empty;
  {
    mutex_lock lock(mu);
    was_empty = events.empty();
    events.emplace_back(event, std::move(on_reached));
  }
  if (was_empty) cv.notify_one();
}

EventManager::ThreadInfo::~ThreadInfo() {
  assert(thread.joinable());
  {
    mutex_lock lock(mu);
    status.store(ThreadStatus::SHUTTING_DOWN, std::memory_order_release);
    // Notify while holding the lock to make sure we don't race with
    // cv.wait(), which is protected with mu.
    cv.notify_one();
  }
  thread.join();
}

void EventManager::Synchronize(stream::Event event, stream::Stream stream,
                               EventReachedCallback on_reached) {
  // Most of the time, threads_ is only read. Consider using a shared_mutex
  // (added in C++17) for mu_ if locking here shows up in profiles.

  // If there is already a thread assigned to `stream`, add `event` to its
  // queue.
  // We use explicit locking instead of mutex_lock to avoid grabbing multiple
  // mutexes at once and worrying about lock ordering. Static analysis is smart
  // enough to catch lock/unlock bugs.
  mu_.lock();
  auto it = threads_.find(stream);
  if (it != threads_.end()) {
    ThreadInfo* ti = it->second.get();
    // Even if another stream steals this ThreadInfo, ThreadInfo will not be
    // deleted. So, it is safe to unlock mu_ here.
    mu_.unlock();
    ti->AddEvent(event, std::move(on_reached));
    return;
  }

  // There is no thread already assigned to `stream`. Look for a thread assigned
  // to another stream and steal it.
  it = std::find_if(
      threads_.begin(), threads_.end(), [](const auto& stream_and_thread) {
        return stream_and_thread.second->status.load(
                   std::memory_order_acquire) == ThreadStatus::IDLE;
      });
  if (it != threads_.end()) {
    mu_.unlock();
    it->second->AddEvent(event, std::move(on_reached));
    return;
  }

  // All threads are busy. Create a new one.
  threads_[stream] = std::make_unique<ThreadInfo>();
  ThreadInfo* new_thread = threads_[stream].get();
  mu_.unlock();
  // Start the thread after releasing mu_ to maintain the invariant
  // that we never hold more than one mutex.
  new_thread->thread = std::thread([new_thread]() { ThreadFn(new_thread); });
  new_thread->AddEvent(event, std::move(on_reached));
}

void EventManager::ThreadFn(ThreadInfo* thread_info) {
  while (true) {
    // Wait for some events to come and copy them into a temp deque.
    std::deque<EventAndCallback> swapped_events;
    bool shutdown_after_processing = false;
    {
      mutex_lock lock(thread_info->mu);
      while (thread_info->events.empty()) {
        ThreadStatus previous = thread_info->status.exchange(
            ThreadStatus::IDLE, std::memory_order_release);
        if (previous == ThreadStatus::SHUTTING_DOWN) {
          return;
        }
        thread_info->cv.wait(lock);
      }
      ThreadStatus previous = thread_info->status.exchange(
          ThreadStatus::PROCESSING, std::memory_order_release);
      if (previous == ThreadStatus::SHUTTING_DOWN) {
        shutdown_after_processing = true;
      }
      swapped_events.swap(thread_info->events);
    }

    // Synchronize events and invoke their callbacks.
    for (EventAndCallback& item : swapped_events) {
      item.on_reached(stream::EventSynchronize(item.event));
    }
    swapped_events.clear();

    if (shutdown_after_processing) {
      return;
    }
  }
}

}  // namespace gpu
}  // namespace tfrt
