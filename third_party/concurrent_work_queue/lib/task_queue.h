// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// TaskQueue is a fixed-size, non-blocking FIFO queue of Task items.
//
// This queue is not strictly lock-free (system as a whole moves forward
// regardless of anything), because if a thread calling PopBack() preempted
// after it updated `back_` before it updated `e->state`, no other thread will
// be able to successfully PushFront() past that element.
//
// Also PopBack() might return empty optional, if a thread calling PushFront()
// was preempted after updating `front_` before it updated `e->state`.
//
// In practice, without large thread oversubscription this queue almost
// never blocks or returns empty result from PopBack() when it's not empty.
//
// Clients should never rely on PopBack() result for strong emptyness check, for
// this purpose Empty() guaranteed to return correct result.
//
// Based on "Bounded MPMC queue":
// http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_QUEUE_H_

#include <array>
#include <atomic>
#include <limits>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include "tfrt/host_context/task_function.h"

namespace tfrt {
namespace internal {

class TaskQueue {
 public:
  static const unsigned kCapacity = 1024;

  static_assert((kCapacity > 2) && (kCapacity <= (64u << 10u)),
                "TaskQueue capacity must be in [4, 65536] range");
  static_assert((kCapacity & (kCapacity - 1)) == 0,
                "TaskQueue capacity must be a power of two for fast masking");

  TaskQueue() : front_(0), back_(0) {
    for (unsigned i = 0; i < kCapacity; ++i) array_[i].state.store(i);
  }

  TaskQueue(const TaskQueue&) = delete;
  void operator=(const TaskQueue&) = delete;

  ~TaskQueue() { assert(Empty()); }

  // PushFront() inserts task at the beginning of the queue.
  //
  // If the queue is full, returns passed in task wrapped in optional, otherwise
  // returns empty optional.
  [[nodiscard]] llvm::Optional<TaskFunction> PushFront(TaskFunction task) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e;

    for (;;) {
      e = &array_[front & kMask];
      unsigned state = e->state.load(std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(state) - static_cast<int64_t>(front);

      // Try to acquire an ownership of element at `front`.
      if (diff == 0 && front_.compare_exchange_strong(
                           front, front + 1, std::memory_order_relaxed)) {
        break;
      }

      // We wrapped around the queue, and have no space in the queue.
      if (diff < 0) return {std::move(task)};

      // Another thread acquired element at `front` index.
      if (diff > 0) front = front_.load(std::memory_order_relaxed);
    }

    // Move the task to the acquired element.
    e->task = std::move(task);
    e->state.store(front + 1, std::memory_order_release);

    return llvm::None;
  }

  // PopBack() removes and returns the last elements in the queue.
  //
  // If the queue is empty returns empty optional.
  [[nodiscard]] llvm::Optional<TaskFunction> PopBack() {
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e;

    for (;;) {
      e = &array_[back & kMask];
      unsigned state = e->state.load(std::memory_order_acquire);
      int64_t diff =
          static_cast<int64_t>(state) - static_cast<int64_t>(back + 1);

      // Element at `back` is ready, try to acquire its ownership.
      if (diff == 0 && back_.compare_exchange_strong(
                           back, back + 1, std::memory_order_relaxed)) {
        break;
      }

      // We've reached an empty element.
      if (diff < 0) return llvm::None;

      // Another thread popped a task from element at 'back'.
      if (diff > 0) back = back_.load(std::memory_order_relaxed);
    }

    TaskFunction task = std::move(e->task);
    e->state.store(back + kCapacity, std::memory_order_release);

    return {std::move(task)};
  }

  bool Empty() const { return Size() == 0; }

  // Size() returns a queue size estimate that potentially could be larger
  // than the real number of tasks in the queue. It never can be smaller.
  unsigned Size() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    unsigned front = front_.load(std::memory_order_acquire);
    for (;;) {
      // Capture a consistent snapshot of front/back.
      unsigned back = back_.load(std::memory_order_acquire);
      unsigned front1 = front_.load(std::memory_order_relaxed);
      if (front != front1) {
        front = front1;
        std::atomic_thread_fence(std::memory_order_acquire);
        continue;
      }

      return std::min(kCapacity, front - back);
    }
  }

  // Delete all the elements from the queue.
  void Flush() {
    while (!Empty()) {
      llvm::Optional<TaskFunction> task = PopBack();
      assert(task.has_value());
    }
  }

 private:
  // Mask for extracting front and back pointer positions.
  static constexpr unsigned kMask = kCapacity - 1;

  struct Elem {
    std::atomic<unsigned> state;
    TaskFunction task;
  };

  // Align to 128 byte boundary to prevent false sharing between front and back
  // pointers, they are accessed from different threads.
  alignas(128) std::atomic<unsigned> front_;
  alignas(128) std::atomic<unsigned> back_;

  std::array<Elem, kCapacity> array_;
};

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_QUEUE_H_
