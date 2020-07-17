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
#include <deque>
#include <thread>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/gpu/stream/dense_map_utils.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace gpu {

// EventManager implements efficient waiting and polling for events.
class EventManager {
 public:
  using EventReachedCallback = llvm::unique_function<void(llvm::Error)>;

  EventManager();

  // Invokes `on_reached` on a thread managed by this EventManager
  // when `event` has been reached. The event must have been recorded
  // on `stream`.
  // `event` and `stream` must outlive a call to `on_reached`.
  // This method is thread-safe.
  //
  // The implementation uses a single cached thread per stream.
  // If the thread is blocked waiting for some event when this method is called,
  // the `event` will be queued and synchronized to after the thread becomes
  // available again. This approach can be suboptimal if the order of event
  // recording is different from the order of calls to this method. This
  // different order can also lead to deadlocks if there are blocking callbacks
  // enqueued on the stream using something like cudaLaunchHostFunc.
  //
  // `on_reached` should generally be lightweight and return quickly, but it is
  // allowed to call EventManager methods.
  //
  // Calling on_reached on the EventManager thread can result in a lot of work
  // being done on that thread if the callback completes some AsyncValue which
  // has AndThen callbacks registered on it. If this becomes a problem, e.g.
  // callbacks are delayed or there is thread contention, users should enqueue
  // the async value completion onto the HostContext's concurrent work queue.
  void Synchronize(stream::Event event, stream::Stream stream,
                   EventReachedCallback on_reached);

 private:
  // Current implementation assigns a single thread for each stream,
  // synchronizes all events of this stream on this thread, and invokes
  // the callbacks on this thread. When we are asked to synchronize on event
  // recorded on a stream that we have not seen before, we first check if any
  // thread assigned to other streams is idle. If so, we steal it by reassigning
  // it to the new stream. If all existing threads are busy, we create a new
  // thread. Because we create a new thread only when all existing ones are
  // busy, the total number of threads is bounded by the maximum number of
  // streams with concurrently active events.
  //
  // Stealing is a kind of garbage collection. Because we are not notified when
  // a stream is destroyed, we don't know when we should reclaim its assigned
  // thread.
  //
  // We are not using the threads of BlockingWorkQueue because:
  //  - All the threads in BlockingWorkQueue can be blocked on lower priority
  //    work. Since waiting for GPU event is usually latency sensitive, we want
  //    to isolate it.
  //  - We would like to not use a thread per event, but a thread per stream.
  //    BlockingWorkQueue does not currently provide a way to achieve this.
  //  - In the future, we might want to provide a Synchronize implementation
  //    based on polling (i.e. periodically calling cuEventQuery over a list of
  //    events). Efficient polling requires a timer mechanism, which
  //    BlockingWorkQueue does not currently provide.

  struct EventAndCallback {
    EventAndCallback(stream::Event event, EventReachedCallback on_reached)
        : event(event), on_reached(std::move(on_reached)) {}
    stream::Event event;
    EventReachedCallback on_reached;
  };

  enum class ThreadStatus {
    PROCESSING,
    IDLE,
    SHUTTING_DOWN,
  };

  struct ThreadInfo {
    ThreadInfo() = default;

    ~ThreadInfo();

    void AddEvent(stream::Event event, EventReachedCallback on_reached);

    std::atomic<ThreadStatus> status;
    // IDEA: If it becomes important consider using a vector or some lock-free
    // queue.
    std::deque<EventAndCallback> events TFRT_GUARDED_BY(mu);
    mutex mu;
    condition_variable cv;
    std::thread thread;
  };

  static void ThreadFn(ThreadInfo* thread_info);

  mutable mutex mu_;
  condition_variable cv_;
  llvm::DenseMap<stream::Stream, std::unique_ptr<ThreadInfo>> threads_
      TFRT_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_EVENT_MANAGER_H_
