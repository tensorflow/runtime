// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- non_blocking_work_queue.h --------------------------------*- C++ -*-===//
//
// Work queue implementation based on non-blocking concurrency primitives
// optimized for CPU intensive non-blocking compute tasks.
//
// All threads managed by this work queue have their own pending tasks queue of
// a fixed size. Thread worker loop tries to pop the next task from the front of
// its own queue first, if the queue is empty it's going into a steal loop, and
// tries to steal the next task from the back of the other thread pending tasks
// queue.
//
// If a thread was not able to find the next task function to execute, it's
// parked on a conditional variable, and waits for the notification from the
// `AddTask()`.
//
// Before parking on a conditional variable, thread might go into a spin loop
// (controlled by `kMaxSpinningThreads` constant), and execute steal loop for a
// fixed number of iterations. This allows to skip expensive park/unpark
// operations, and reduces latency. Increasing `kMaxSpinningThreads` improves
// latency at the cost of burned CPU cycles.
//
// Work queue implementation is parametrized by `ThreadingEnvironment` that
// allows to provide custom thread implementation:
//
//  struct ThreadingEnvironment {
//    // Type alias for the underlying thread implementation.
//    using Thread = ...
//
//    // Starts a new thread running function `f` with arguments `arg`.
//    template <class Function, class... Args>
//    std::unique_ptr<Thread> StartThread(Function&& f, Args&&... args) { ... }
//
//    // Blocks the current thread until the `thread` finishes its execution.
//    static void Join(Thread* thread) { ... }
//
//    // Separates the thread of execution from the thread object.
//    static void Detach(Thread* thread) { ... }
//
//    // Returns current thread id hash code.
//    static uint64_t ThisThreadIdHash() {... }
//  }
//
// Work stealing algorithm is based on:
//
//   "Thread Scheduling for Multiprogrammed Multiprocessors"
//   Nimar S. Arora, Robert D. Blumofe, C. Greg Plaxton
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_

#include <memory>
#include <string>
#include <thread>

#include "event_count.h"
#include "task_deque.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace internal {

struct FastRng {
  constexpr explicit FastRng(uint64_t state) : state(state) {}

  unsigned operator()() {
    uint64_t current = state;
    // Update the internal state
    state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22u)) >>
                                 (22 + (current >> 61u)));
  }

  uint64_t state;
};

template <typename ThreadingEnvironment>
class NonBlockingWorkQueue {
  using EventCount = ::tfrt::internal::EventCount;
  using Queue = ::tfrt::internal::TaskDeque;

  using Thread = typename ThreadingEnvironment::Thread;

 public:
  explicit NonBlockingWorkQueue(int num_threads);
  ~NonBlockingWorkQueue();

  void AddTask(TaskFunction task);

  // Quiesce blocks caller thread until all submitted tasks are completed and
  // all worker threads are in the parked state.
  void Quiesce();

  // Steal() tries to steal task from any worker managed by this queue. Returns
  // llvm::None if it was not able to find task to steal.
  LLVM_NODISCARD llvm::Optional<TaskFunction> Steal();

  // Returns `true` if all worker threads are parked. This is a weak signal of
  // work queue emptyness, because worker thread might be notified, but not
  // yet unparked and running. For strong guarantee must use use Quiesce.
  bool AllBlocked() const { return NumBlockedThreads() == num_threads_; }

  // Returns current thread id if the caller thread is managed by `this`,
  // returns `-1` otherwise.
  int CurrentThreadId() const;

  // CheckCallerThread() will abort the program if the caller thread is managed
  // by `*this`. This is required to prevent deadlocks from calling `Quiesce`
  // from a thread managed by the current worker queue.
  void CheckCallerThread(const char* function_name) const;

  // Stop all threads managed by this work queue.
  void Cancel();

 private:
  // TODO(ezhulenev): Make this a runtime parameter? More spinning threads help
  // to reduce latency at the cost of wasted CPU cycles.
  static constexpr int kMaxSpinningThreads = 1;

  // The number of steal loop spin iterations before parking (this number is
  // divided by the number of threads, to get spin count for each thread).
  static constexpr int kSpinCount = 5000;

  // If there are enough active threads with an empty pending task queues, there
  // is no need for spinning before parking a thread that is out of work to do,
  // because these active threads will go into a steal loop after finishing with
  // their current tasks.
  //
  // In the worst case when all active threads are executing long/expensive
  // tasks, the next AddTask() will have to wait until one of the parked threads
  // will be unparked, however this should be very rare in practice.
  static constexpr int kMinActiveThreadsToStartSpinning = 4;

  struct PerThread {
    constexpr PerThread() : parent(nullptr), rng(0), thread_id(-1) {}
    NonBlockingWorkQueue* parent;
    FastRng rng;    // Random number generator
    int thread_id;  // Worker thread index in the workers queue
  };

  struct ThreadData {
    ThreadData() : thread(), queue() {}
    std::unique_ptr<Thread> thread;
    Queue queue;
  };

  // Main worker thread loop.
  void WorkerLoop(int thread_id);

  // WaitForWork() blocks until new work is available (returns true), or if it
  // is time to exit (returns false). Can optionally return a task to execute in
  // `task` (in such case `task.hasValue() == true` on return).
  LLVM_NODISCARD bool WaitForWork(EventCount::Waiter* waiter,
                                  llvm::Optional<TaskFunction>* task);

  // NonEmptyQueueIndex() returns the index of a non-empty worker queue, or `-1`
  // if all queues are empty.
  LLVM_NODISCARD int NonEmptyQueueIndex();

  LLVM_NODISCARD static PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  // StartSpinning() checks if the number of threads in the spin loop is less
  // than the allowed maximum, if so increments the number of spinning threads
  // by one and returns true (caller must enter the spin loop). Otherwise
  // returns false, and the caller must not enter the spin loop.
  LLVM_NODISCARD bool StartSpinning();

  // StopSpinning() decrements the number of spinning threads by one. It also
  // checks if there were any tasks submitted into the pool without notifying
  // parked threads, and decrements the count by one. Returns true if the number
  // of tasks submitted without notification was decremented, in this case
  // caller thread might have to call Steal() one more time.
  LLVM_NODISCARD bool StopSpinning();

  // IsNotifyParkedThreadRequired() returns true if parked thread must be
  // notified about new added task. If there are threads spinning in the steal
  // loop, there is no need to unpark any of the waiting threads, the task will
  // be picked up by one of the spinning threads.
  LLVM_NODISCARD bool IsNotifyParkedThreadRequired();

  unsigned NumBlockedThreads() const { return blocked_.load(); }
  unsigned NumActiveThreads() const { return num_threads_ - blocked_.load(); }

  int num_threads_;
  ThreadingEnvironment threading_environment_;

  std::vector<ThreadData> thread_data_;
  std::vector<unsigned> coprimes_;

  std::atomic<unsigned> blocked_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;

  // Use a conditional variable to notify waiters when all worker threads are
  // blocked. This is used to park caller thread in Quiesce() when there is no
  // work to steal, but not all threads are parked.
  mutex all_blocked_mu_;
  condition_variable all_blocked_cv_;

  // Spinning state layout:
  // - Low 32 bits encode the number of threads that are spinning in steal loop.
  //
  // - High 32 bits encode the number of tasks that were submitted to the pool
  //   without a call to event_count_.Notify(). This number can't be larger than
  //   the number of spinning threads. Each spinning thread, when it exits the
  //   spin loop must check if this number is greater than zero, and maybe make
  //   another attempt to steal a task and decrement it by one.
  static constexpr uint64_t kNumSpinningBits = 32;
  static constexpr uint64_t kNumSpinningMask = (1ull << kNumSpinningBits) - 1;
  static constexpr uint64_t kNumNoNotifyBits = 32;
  static constexpr uint64_t kNumNoNotifyShift = 32;
  static constexpr uint64_t kNumNoNotifyMask = ((1ull << kNumNoNotifyBits) - 1)
                                               << kNumNoNotifyShift;
  std::atomic<uint64_t> spinning_state_;

  struct SpinningState {
    uint64_t num_spinning;         // number of spinning threads
    uint64_t num_no_notification;  // number of tasks submitted without
                                   // notifying waiting threads

    // Decode `spinning_state_` value.
    static SpinningState Decode(uint64_t state) {
      uint64_t num_spinning = (state & kNumSpinningMask);
      uint64_t num_no_notification =
          (state & kNumNoNotifyMask) >> kNumNoNotifyShift;

      assert(num_no_notification <= num_spinning);

      return {num_spinning, num_no_notification};
    }

    // Encode as `spinning_state_` value.
    uint64_t Encode() const {
      return (num_no_notification << kNumNoNotifyShift) | num_spinning;
    }
  };

  // Reduce `x` into [0, size) range (compute `x % size`).
  // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
  uint32_t FastReduce(uint32_t x, uint32_t size) {
    return (static_cast<uint64_t>(x) * static_cast<uint64_t>(size)) >> 32u;
  }

  EventCount event_count_;
};

// Calculate coprimes of all numbers [1, n].
//
// Coprimes are used for random walks over all threads in Steal
// and NonEmptyQueueIndex. Iteration is based on the fact that if we take a
// random starting thread index `t` and calculate `num_threads - 1` subsequent
// indices as `(t + coprime) % num_threads`, we will cover all threads without
// repetitions (effectively getting a pseudo-random permutation of thread
// indices).
inline std::vector<unsigned> ComputeCoprimes(int n) {
  std::vector<unsigned> coprimes;
  for (unsigned i = 1; i <= n; i++) {
    unsigned a = i;
    unsigned b = n;
    // If GCD(a, b) == 1, then a and b are coprimes.
    while (b != 0) {
      unsigned tmp = a;
      a = b;
      b = tmp % b;
    }
    if (a == 1) coprimes.push_back(i);
  }
  return coprimes;
}

inline void Execute(llvm::Optional<TaskFunction>& task) {
  assert(task.hasValue());
  task.getValue()();
}

template <typename ThreadingEnvironment>
NonBlockingWorkQueue<ThreadingEnvironment>::NonBlockingWorkQueue(
    int num_threads)
    : num_threads_(num_threads),
      thread_data_(num_threads_),
      coprimes_(ComputeCoprimes(num_threads_)),
      blocked_(0),
      done_(false),
      cancelled_(false),
      spinning_state_(0),
      event_count_(num_threads_) {
  assert(num_threads_ >= 1);
  for (int i = 0; i < num_threads_; i++) {
    thread_data_[i].thread =
        threading_environment_.StartThread([this, i]() { WorkerLoop(i); });
  }
}

template <typename ThreadingEnvironment>
NonBlockingWorkQueue<ThreadingEnvironment>::~NonBlockingWorkQueue() {
  done_ = true;

  // Now if all threads block without work, they will start exiting.
  // But note that threads can continue to work arbitrary long,
  // block, submit new work, unblock and otherwise live full life.
  if (!cancelled_) {
    event_count_.Notify(true);
  } else {
    // Since we were cancelled, there might be entries in the queues.
    // Empty them to prevent their destructor from asserting.
    for (ThreadData& thread_data : thread_data_) {
      thread_data.queue.Flush();
    }
  }
  // Join to all worker threads before calling destructors.
  for (ThreadData& thread_data : thread_data_) {
    ThreadingEnvironment::Join(thread_data.thread.get());
    thread_data.thread.reset();
  }
}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::AddTask(TaskFunction task) {
  // If the worker queue is full, we will execute `task` in the current thread.
  llvm::Optional<TaskFunction> inline_task;

  // If a caller thread is managed by `this` we push the new task into the front
  // of thread own queue (LIFO execution order). PushFront is completely lock
  // free (PushBack requires a mutex lock), and improves data locality (in
  // practice tasks submitted together share some data).
  //
  // If a caller is a free-standing thread (or worker of another pool), we push
  // the new task into a random queue (FIFO execution order). Tasks still could
  // be executed in LIFO order, if they would be stolen by other workers.

  PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    // Worker thread of this pool, push onto the thread's queue.
    Queue& q = thread_data_[pt->thread_id].queue;
    inline_task = q.PushFront(std::move(task));
  } else {
    // A free-standing thread (or worker of another pool).
    unsigned rnd = FastReduce(pt->rng(), num_threads_);
    Queue& q = thread_data_[rnd].queue;
    inline_task = q.PushBack(std::move(task));
  }
  // Note: below we touch this after making w available to worker threads.
  // Strictly speaking, this can lead to a racy-use-after-free. Consider that
  // Schedule is called from a thread that is neither main thread nor a worker
  // thread of this pool. Then, execution of w directly or indirectly
  // completes overall computations, which in turn leads to destruction of
  // this. We expect that such scenario is prevented by program, that is,
  // this is kept alive while any threads can potentially be in Schedule.
  if (!inline_task.hasValue()) {
    if (IsNotifyParkedThreadRequired()) event_count_.Notify(false);
  } else {
    Execute(inline_task);  // Push failed, execute directly.
  }
}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::CheckCallerThread(
    const char* function_name) const {
  PerThread* pt = GetPerThread();
  TFRT_LOG_IF(FATAL, pt->parent == this)
      << "Error at " << __FILE__ << ":" << __LINE__ << ": " << function_name
      << " should not be called by a work thread already managed by the queue.";
}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::Quiesce() {
  CheckCallerThread("NonBlockingWorkQueue::Quiesce");

  // Keep stealing tasks until we reach a point when we have nothing to steal
  // and all worker threads are in blocked state.
  llvm::Optional<TaskFunction> task = Steal();

  while (task.hasValue()) {
    // Execute stolen task in the caller thread.
    Execute(task);

    // Try to steal the next task.
    task = Steal();
  }

  // If we didn't find a task to execute, and there are still worker threads
  // running, park current thread on a conditional variable until all worker
  // threads are blocked.
  if (!AllBlocked()) {
    mutex_lock lock(all_blocked_mu_);
    all_blocked_cv_.wait(
        lock, [this]() TFRT_REQUIRES(all_blocked_mu_) { return AllBlocked(); });
  }
}

template <typename ThreadingEnvironment>
LLVM_NODISCARD llvm::Optional<TaskFunction>
NonBlockingWorkQueue<ThreadingEnvironment>::Steal() {
  PerThread* pt = GetPerThread();
  unsigned r = pt->rng();
  unsigned victim = FastReduce(r, num_threads_);
  unsigned inc = coprimes_[FastReduce(r, coprimes_.size())];

  for (unsigned i = 0; i < num_threads_; i++) {
    llvm::Optional<TaskFunction> t = thread_data_[victim].queue.PopBack();
    if (t.hasValue()) return t;

    victim += inc;
    if (victim >= num_threads_) {
      victim -= num_threads_;
    }
  }
  return llvm::None;
}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::WorkerLoop(int thread_id) {
  PerThread* pt = GetPerThread();
  pt->parent = this;
  pt->rng = FastRng(ThreadingEnvironment::ThisThreadIdHash());
  pt->thread_id = thread_id;

  Queue& q = thread_data_[thread_id].queue;
  EventCount::Waiter* waiter = event_count_.waiter(thread_id);

  // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
  // proportional to num_threads_ and we assume that new work is scheduled at
  // a constant rate, so we set spin_count to 5000 / num_threads_. The
  // constant was picked based on a fair dice roll, tune it.
  const int spin_count = num_threads_ > 0 ? kSpinCount / num_threads_ : 0;

  while (!cancelled_) {
    llvm::Optional<TaskFunction> t = q.PopFront();
    if (!t.hasValue()) {
      t = Steal();
      if (!t.hasValue()) {
        // Maybe leave thread spinning. This reduces latency.
        const bool start_spinning = StartSpinning();
        if (start_spinning) {
          for (int i = 0; i < spin_count && !t.hasValue(); ++i) {
            t = Steal();
          }

          const bool stopped_spinning = StopSpinning();
          // If a task was submitted to the queue without a call to
          // `event_count_.Notify()`, and we didn't steal anything above, we
          // must try to steal one more time, to make sure that this task will
          // be executed. We will not necessarily find it, because it might have
          // been already stolen by some other thread.
          if (stopped_spinning && !t.hasValue()) {
            t = Steal();
          }
        }

        if (!t.hasValue()) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
      }
    }
    if (t.hasValue()) {
      Execute(t);
    }
  }
}

template <typename ThreadingEnvironment>
bool NonBlockingWorkQueue<ThreadingEnvironment>::WaitForWork(
    EventCount::Waiter* waiter, llvm::Optional<TaskFunction>* task) {
  assert(!task->hasValue());
  // We already did best-effort emptiness check in Steal, so prepare for
  // blocking.
  event_count_.Prewait();
  // Now do a reliable emptiness check.
  int victim = NonEmptyQueueIndex();
  if (victim != -1) {
    event_count_.CancelWait();
    if (cancelled_) {
      return false;
    } else {
      *task = thread_data_[victim].queue.PopBack();
      return true;
    }
  }
  // Number of blocked threads is used as termination condition.
  // If we are shutting down and all worker threads blocked without work,
  // that's we are done.
  blocked_.fetch_add(1);

  // Notify threads that are waiting for "all blocked" event.
  if (blocked_.load() == static_cast<unsigned>(num_threads_)) {
    mutex_lock lock(all_blocked_mu_);
    all_blocked_cv_.notify_all();
  }

  // Prepare to shutdown worker thread if done.
  if (done_ && blocked_.load() == static_cast<unsigned>(num_threads_)) {
    event_count_.CancelWait();
    // Almost done, but need to re-check queues.
    // Consider that all queues are empty and all worker threads are preempted
    // right after incrementing blocked_ above. Now a free-standing thread
    // submits work and calls destructor (which sets done_). If we don't
    // re-check queues, we will exit leaving the work unexecuted.
    if (NonEmptyQueueIndex() != -1) {
      // Note: we must not pop from queues before we decrement blocked_,
      // otherwise the following scenario is possible. Consider that instead
      // of checking for emptiness we popped the only element from queues.
      // Now other worker threads can start exiting, which is bad if the
      // work item submits other work. So we just check emptiness here,
      // which ensures that all worker threads exit at the same time.
      blocked_.fetch_sub(1);
      return true;
    }
    // Reached stable termination state.
    event_count_.Notify(true);
    return false;
  }

  event_count_.CommitWait(waiter);
  blocked_.fetch_sub(1);
  return true;
}

template <typename ThreadingEnvironment>
bool NonBlockingWorkQueue<ThreadingEnvironment>::StartSpinning() {
  if (NumActiveThreads() > kMinActiveThreadsToStartSpinning) return false;

  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    if ((state.num_spinning - state.num_no_notification) >= kMaxSpinningThreads)
      return false;

    // Increment the number of spinning threads.
    ++state.num_spinning;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return true;
    }
  }
}

template <typename ThreadingEnvironment>
bool NonBlockingWorkQueue<ThreadingEnvironment>::StopSpinning() {
  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    // Decrement the number of spinning threads.
    --state.num_spinning;

    // Maybe decrement the number of tasks submitted without notification.
    bool has_no_notify_task = state.num_no_notification > 0;
    if (has_no_notify_task) --state.num_no_notification;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return has_no_notify_task;
    }
  }
}

template <typename ThreadingEnvironment>
bool NonBlockingWorkQueue<
    ThreadingEnvironment>::IsNotifyParkedThreadRequired() {
  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    // If the number of tasks submitted without notifying parked threads is
    // equal to the number of spinning threads, we must wake up one of the
    // parked threads.
    if (state.num_no_notification == state.num_spinning) return true;

    // Increment the number of tasks submitted without notification.
    ++state.num_no_notification;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return false;
    }
  }
}

template <typename ThreadingEnvironment>
int NonBlockingWorkQueue<ThreadingEnvironment>::NonEmptyQueueIndex() {
  PerThread* pt = GetPerThread();
  unsigned r = pt->rng();
  unsigned inc = num_threads_ == 1 ? 1 : coprimes_[r % coprimes_.size()];
  unsigned victim = FastReduce(r, num_threads_);
  for (unsigned i = 0; i < num_threads_; i++) {
    if (!thread_data_[victim].queue.Empty()) {
      return static_cast<int>(victim);
    }
    victim += inc;
    if (victim >= num_threads_) {
      victim -= num_threads_;
    }
  }
  return -1;
}

template <typename ThreadingEnvironment>
int NonBlockingWorkQueue<ThreadingEnvironment>::CurrentThreadId() const {
  const PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    return pt->thread_id;
  } else {
    return -1;
  }
}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::Cancel() {
  cancelled_ = true;
  done_ = true;

  // Wake up the threads without work to let them exit on their own.
  event_count_.Notify(true);
}

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_
