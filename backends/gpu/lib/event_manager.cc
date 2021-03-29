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

#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/thread_environment.h"

namespace tfrt {
namespace gpu {

EventManager::EventManager(HostContext& host_context)
    : host_context_(host_context),
      worker_cancelled_(false),
      worker_(ThreadingEnvironment::StartThread(
          "tfrt-event-manager", &EventManager::PollEvents, this)) {}

AsyncValueRef<Chain> EventManager::Synchronize(RCReference<RcEvent> event) {
  // Check if the event is already ready.
  auto query_result = stream::EventQuery(event->get());
  if (!query_result) {
    return MakeErrorAsyncValueRef(&host_context_,
                                  StrCat("EventManager error querying event: ",
                                         query_result.takeError()));
  }
  if (*query_result) {
    return GetReadyChain(&host_context_);
  }

  auto pending_async_value_ref =
      MakeUnconstructedAsyncValueRef<Chain>(&host_context_);
  // Add Ref outside of mutex. In order to bitpack event record, we have a
  // pointer to the AsyncValue instead of an AsyncValueRef.
  AsyncValue* pending_async_value =
      pending_async_value_ref.GetAsyncValue()->AddRef();

  events_mutex_.lock();
  events_.push_back(EventRecord{{pending_async_value, RecordStatus::kPending},
                                std::move(event)});
  condition_.notify_one();
  events_mutex_.unlock();
  return pending_async_value_ref;
}

void EventManager::PollEvents() {
  while (!worker_cancelled_.load(std::memory_order_relaxed)) {
    {
      mutex_lock lock(events_mutex_);
      condition_.wait(lock, [&]() TFRT_REQUIRES(events_mutex_) -> bool {
        return !events_.empty() ||
               worker_cancelled_.load(std::memory_order_relaxed);
      });
    }

    bool has_events = true;
    while (has_events) {
      if (worker_cancelled_.load(std::memory_order_relaxed)) {
        TFRT_LOG_INFO << "EventManager worker thread cancelled.";
        return;
      }

      std::vector<EventRecord> to_resolve;
      {
        mutex_lock lock(events_mutex_);
        for (auto& event_record : events_) {
          auto query_result = stream::EventQuery(event_record.event->get());
          if (!query_result) {
            // Report errors immediately
            event_record.pending_async_value.getPointer()->SetError(
                DecodedDiagnostic(query_result.takeError()));
            event_record.pending_async_value.getPointer()->DropRef();
            event_record.pending_async_value.setInt(RecordStatus::kError);
            continue;
          }
          // The event is ready.
          if (*query_result) {
            event_record.pending_async_value.setInt(RecordStatus::kResolved);
          }
        }
        // Partition events_ such that the head of the list contains the
        // kPending events, and the tail contains a mix of kErrors and
        // kResolved. We'll remove all non-pending event records.
        auto to_resolve_it = std::partition(
            events_.begin(), events_.end(), [](const auto& record) {
              return record.pending_async_value.getInt() ==
                     RecordStatus::kPending;
            });
        // Use remove_if to partition the vector again such that all the
        // kResolved are clustered together.
        auto to_remove_it = std::remove_if(
            to_resolve_it, events_.end(), [](const auto& record) {
              return record.pending_async_value.getInt() ==
                     RecordStatus::kError;
            });

        // Move the kDone events into a local vector so we can resolve them
        // outside the mutex.
        to_resolve.insert(to_resolve.end(),
                          std::make_move_iterator(to_resolve_it),
                          std::make_move_iterator(to_remove_it));
        events_.erase(to_resolve_it, events_.end());
        has_events = !events_.empty();
      }

      if (!to_resolve.empty()) {
        EnqueueWork(&host_context_, [events = std::move(to_resolve)] {
          for (auto& record : events) {
            record.pending_async_value.getPointer()->emplace<Chain>();
            record.pending_async_value.getPointer()->DropRef();
          }
        });
      }
      std::this_thread::yield();
    }
  }
  TFRT_LOG_INFO << "Exiting EventManager worker thread.";
}

EventManager::~EventManager() {
  worker_cancelled_.store(true, std::memory_order_relaxed);
  {
    mutex_lock lock(events_mutex_);
    condition_.notify_one();
  }
  // Joins thread on destruction.
  worker_.reset();

  mutex_lock lock(events_mutex_);
  TFRT_LOG_INFO << "EventManager destroyed with " << events_.size()
                << " pending events.";
  for (auto& event : events_) {
    event.pending_async_value.getPointer()->SetError(
        DecodedDiagnostic("Cancelled EventManager."));
    event.pending_async_value.getPointer()->DropRef();
  }
}

}  // namespace gpu
}  // namespace tfrt
