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

//===- event_manager.h - Utility for CUDA/HIP event waiting  ----*- C++ -*-===//
//
// This file declares the EventManager class that can be used to get
// notification on the host when some GPU events have been reached.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_EVENT_MANAGER_H_
#define TFRT_GPU_EVENT_MANAGER_H_

#include <atomic>
#include <thread>

#include "llvm/ADT/PointerIntPair.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"
#include "tfrt/support/thread_environment.h"

namespace tfrt {

class Chain;

namespace gpu {

// A ref-counted owned GPU event.
class RcEvent : public ReferenceCounted<RcEvent> {
 public:
  explicit RcEvent(stream::OwningEvent event) : event_(std::move(event)) {}

  stream::Event get() const { return event_.get(); }

 private:
  stream::OwningEvent event_;
};

// EventManager implements efficient waiting and polling for events. The
// EventManager maintains a single thread dedicated to event polling.
class EventManager {
 public:
  explicit EventManager(HostContext& host_context);
  ~EventManager();

  // Returns an AsyncValueRef which will become available once the provided
  // event has been reported. The AsyncValueRef will be constructed within a
  // Task in the ConcurrentWorkQueue, so standard async considerations (i.e. no
  // blocking within a CWQ task) apply.
  AsyncValueRef<Chain> Synchronize(RCReference<RcEvent> event);

 private:
  // Worker thread function. If there are events in the events_ queue, queries
  // for their completion. Otherwise sleeps until there are new events.
  void PollEvents();

  // Identifies the status of the pending_async_value in an EventRecod without
  // the need for querying the internal AsyncValue atomics.
  enum class RecordStatus {
    kPending,
    kResolved,
    kError,
  };

  struct EventRecord {
    llvm::PointerIntPair<AsyncValue*, 2, RecordStatus> pending_async_value;
    RCReference<RcEvent> event;
  };

  HostContext& host_context_;

  mutex events_mutex_;
  condition_variable condition_;
  std::vector<EventRecord> events_ TFRT_GUARDED_BY(events_mutex_);

  std::atomic_bool worker_cancelled_;
  // TODO(imintz): Evaluate polling on the non-blocking work queue with a task
  // that reschedules itself.
  std::unique_ptr<ThreadingEnvironment::Thread> worker_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_EVENT_MANAGER_H_
