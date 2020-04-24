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

//===- concurrent_work_queue.h - Work Queue Abstraction ---------*- C++ -*-===//
//
// This file declares the generic interface for concurrent work queue
// abstractions.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_CONCURRENT_WORK_QUEUE_H_
#define TFRT_HOST_CONTEXT_CONCURRENT_WORK_QUEUE_H_

#include <functional>
#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
class AsyncValue;

// This is a pure virtual base class for concurrent work queue implementations.
// This provides an abstraction for adding work items to a queue to be executed
// later. Items are executed in a generally FIFO order, but can also be executed
// in parallel across multiple worker threads.
//
// Note that this is not a generic "thread pool" - it is intentionally limited.
// It accepts two types of tasks, blocking tasks and non-blocking
// (compute-bound) tasks.
//
// The requirements for non-blocking/compute bound task items are:
//  * The items are implicitly parallel and require no locking or other
//    synchronization between them.
//  * The items never call blocking APIs that block the kernel thread they are
//    running on.  They may perform arbitrary computation and enqueue other
//    work though.
//  * The workitems *are* allowed to take mutexes guarding short regions, even
//    though they can technically block when under contention.
//
// The requirements for blocking items are:
//  * The items should spend most of their time waiting for external events,
//    such as file IO and event queues and should have minimum amount of
//    compute.
//
// Adding a blocking task to the queue might fail (if implementation chose
// fixed size internal queue for storing pending tasks) and a caller must decide
// what to do with such tasks. Unlike the non-blocking task, caller thread is
// not allowed to execute blocking tasks, because the caller potentially could
// be running inside a thread-pool that runs non-blocking CPU intensive work.
//
// This class and its subclasses are non-copyable and non-movable.  This may be
// subclassed to provide domain specific implementations, but should not be used
// directly - instead, you should interact with HostContext.
class ConcurrentWorkQueue {
 public:
  // Undefined what happens to pending work when destructor is called.
  virtual ~ConcurrentWorkQueue();

  // Return a human-readable description of the work queue.
  virtual std::string name() const = 0;

 protected:
  // Enqueue a block of work. Thread-safe.
  virtual void AddTask(TaskFunction work) = 0;

  // Enqueue a blocking task. Thread-safe
  // Return empty optional if the work is enqueued successfully, otherwise,
  // returns the argument wrapped in an optional.
  LLVM_NODISCARD virtual Optional<TaskFunction> AddBlockingTask(
      TaskFunction work) = 0;

  // Block until the specified values are available (either with a value or an
  // error result).
  //
  // This should not be called by a thread managed by the work queue.
  virtual void Await(ArrayRef<RCReference<AsyncValue>> values) = 0;

  // Block until the system is quiescent (no pending work and no inflight work).
  //
  // This should not be called by a thread managed by the work queue.
  virtual void Quiesce() = 0;

  // Return the number of parallel tasks maintained by this work queue.
  // Kernels can use this as a hint indicating the maximum useful number of
  // work items they should break themselves into - e.g. zero means it is best
  // to run on the currently active thread and enqueue no work items.
  // TODO(clattner): this is a terrible name.
  virtual int GetParallelismLevel() const = 0;

  ConcurrentWorkQueue() = default;

 private:
  // Clients should access ConcurrentWorkQueue's methods via HostContext.
  friend class HostContext;
  // Tests-only.
  friend class ConcurrentWorkQueueTestHelper;

  ConcurrentWorkQueue(const ConcurrentWorkQueue&) = delete;
  ConcurrentWorkQueue& operator=(const ConcurrentWorkQueue&) = delete;
};

// Create a thread pool that only uses the host donor thread, involving no
// synchronization.
std::unique_ptr<ConcurrentWorkQueue> CreateSingleThreadedWorkQueue();

// Create a multi-threaded non-blocking thread pool that supports both blocking
// and non-blocking workloads.
//
// Arguments:
//
// num_threads: Number of pre-allocated threads used in non-blocking concurrent
// work queue, in addition to the host donor threads.
//
// max_num_pending_blocking_tasks: Max number of pending blocking tasks accepted
// by the blocking concurrent work queue.
//
// Requires `num_threads` > 0 and `max_num_pending_block_tasks` > 0.
std::unique_ptr<ConcurrentWorkQueue> CreateMultiThreadedWorkQueue(
    int num_threads, int max_blocking_work_queue_threads);

// A factory function for creating ConcurrentWorkQueue objects. The factory
// function defines the semantics of the argument string.
// TODO(pgavin): Consider using a configuration object or other data structure
// instead of parsing a string.
using WorkQueueFactory =
    std::function<std::unique_ptr<ConcurrentWorkQueue>(string_view arg)>;

// Register the given factory for creating work queues with the given name.
// This macro should not be used from a header file.
#define TFRT_WORK_QUEUE_FACTORY(NAME, FACTORY) \
  TFRT_WORK_QUEUE_FACTORY_((NAME), (FACTORY), __COUNTER__)
#define TFRT_WORK_QUEUE_FACTORY_(NAME, FACTORY, N) \
  TFRT_WORK_QUEUE_FACTORY__(NAME, FACTORY, N)
#define TFRT_WORK_QUEUE_FACTORY__(NAME, FACTORY, N)              \
  static bool tfrt_work_queue_factory_##N##_registered_ = []() { \
    ::tfrt::RegisterWorkQueueFactory(NAME, FACTORY);             \
    return true;                                                 \
  }()

// Register a ConcurrentWorkQueueFactory under the given name.  Do not call this
// function directly; use the TFRT_WORK_QUEUE_FACTORY macro instead.
void RegisterWorkQueueFactory(string_view name, WorkQueueFactory factory);

// Create a ConcurrentWorkQueue using the given config string. The config string
// may be the name under which the factory was registered, or may be of the
// format "name:arg", where name is an arbitrary string that will be passed to
// the factory function.
std::unique_ptr<ConcurrentWorkQueue> CreateWorkQueue(string_view config);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_CONCURRENT_WORK_QUEUE_H_
