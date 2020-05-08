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

//===- io.h -----------------------------------------------------*- C++ -*-===//
//
// Helper classes for building IO iterators.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_IO_H_
#define TFRT_LIB_DATA_IO_H_

#include <atomic>
#include <memory>
#include <queue>

#include "dataset.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {
namespace io {

// The goal of a prefetching iterator is to move all slow and potentially
// blocking IO operations out of the Iterator::GetNext() execution path.
//
// Prefetching iterator base class helps writing iterators that read data from
// IO sources (e.g. files on disk). Typical IO source is not thread safe and
// slow. To be able to overlap IO with computations, prefetching iterator
// launches asynchronous tasks, that read elements into the memory buffer ahead
// of the calls to GetNext().
//
// PrefetchingIterator::GetNext() will try to return the next element from the
// prefetch buffer. If the buffer is empty, it will read directly from the
// underlying IO source.
//
// Prefetching iterator uses locks to guarantee that access to the IO source
// is properly synchronized. Derived iterators do not need any additional
// synchronization.
//
// In the common case asynchronous prefetch tasks should run ahead of GetNext(),
// and all calls to get the next element should produce results instantaneously.
//
// This is an internal implementation detail, and it is not exposed to the end
// user as a dataset type.
//
// TODO(ezhulenev): Add flexible prefetching policy based not only on the number
// of prefetched records, but also on the memory consumption.
template <typename ValueType>
class PrefetchingIterator : public Iterator<ValueType> {
 public:
  PrefetchingIterator(int32_t max_num_prefetch_elements,
                      int32_t prefetch_threshold)
      : Iterator<ValueType>(),
        max_num_prefetch_elements_(max_num_prefetch_elements),
        prefetch_threshold_(prefetch_threshold),
        prefetch_enqueued_(0),
        cancel_(false) {
    assert(prefetch_threshold >= 0);
    assert(prefetch_threshold <= max_num_prefetch_elements);
  }

  // Gets the next element from a prefetch buffer, and maybe launches an
  // asyncrhonous prefetch task to fill up the buffer. If the buffer is
  // empty, reads next element from the derived iterator.
  IterationResult<ValueType> GetNext(const ExecutionContext& exec_ctx) final;

 protected:
  // Reads the next element from the underlying IO source. Prefetching iterator
  // guarantees that all calls to this function will be properly synchronized.
  virtual IterationResult<ValueType> GetNextElement(
      const ExecutionContext& exec_cxt) TFRT_REQUIRES(input_mu_) = 0;

 private:
  // Cancels all outstanding asynchonous prefetch tasks.
  void Cancel() { cancel_.store(true, std::memory_order_relaxed); }
  bool IsCancelled() const { return cancel_.load(std::memory_order_relaxed); }

  // State mutext guards access to prefetch buffer and prefetch state. It
  // synchronizes concurrent access between PrefetchingIterator::GetNext()
  // and asynchronous prefetch tasks.
  mutex state_mu_;

  // Input mutex guards non thread safe IO operations implemented by the
  // derived iterator. In practice if asynchronous prefetch tasks are running
  // ahead of PrefetchingIterator::GetNext(), it almost never will be a point
  // of contention.
  mutex input_mu_;

  std::queue<IterationResult<ValueType>> buffer_ TFRT_GUARDED_BY(state_mu_);

  const int32_t max_num_prefetch_elements_;
  const int32_t prefetch_threshold_;

  // Number of pending async prefetched elements.
  int32_t prefetch_enqueued_ TFRT_GUARDED_BY(state_mu_);

  // A flag to cancel all pending async prefetch tasks.
  std::atomic<bool> cancel_;
};

template <typename ValueType>
IterationResult<ValueType> PrefetchingIterator<ValueType>::GetNext(
    const ExecutionContext& exec_ctx) {
  // Code that needs to hold both locks (input and state) must do the
  // locking in the same order to avoid deadlocks:
  //
  //   1. input_mu_
  //   2. state_mu_
  //
  // Asynchronous prefetch tasks and this function follow this rule.
  const int32_t max_prefetch = max_num_prefetch_elements_;
  const int32_t threshold = prefetch_threshold_;

  {
    mutex_lock state_lock(state_mu_);

    // Number of prefetched elements + pending prefetches.
    const int32_t prefetched = buffer_.size() + prefetch_enqueued_;

    // Enqueue a prefetch task if the number of outstanding prefetches falls
    // below a threshold.
    if (prefetched <= threshold && !IsCancelled()) {
      const int32_t prefetch = max_prefetch - prefetched;

      auto task = [iterator = FormRef(this), exec_ctx, prefetch]() {
        mutex_lock input_lock(iterator->input_mu_);
        if (iterator->IsCancelled()) return;

        // IDEA(ezhulenev): Verify these ideas in benchmarks:
        // (1) Instead of grabbing the input lock to read all `prefetch`
        //     elements, we can grab it to read mini-batches, so we would
        //     allow concurrent GetNext() caller to make progress sooner in
        //     case it also reaches GetNextLocked().
        // (2) Grab state lock to push multiple prefetched elements at a
        //     time.
        for (int32_t i = 0; i < prefetch; ++i) {
          auto next = iterator->GetNextElement(exec_ctx);
          bool cancel =
              internal::IsConcreteAndEmpty(next) || next.eof.IsError();
          {
            mutex_lock state_lock(iterator->state_mu_);
            iterator->buffer_.push(std::move(next));
            iterator->prefetch_enqueued_--;
          }
          if (cancel) {
            iterator->Cancel();
            break;
          }
        }
      };

      if (exec_ctx.host()->EnqueueBlockingWork(std::move(task)))
        prefetch_enqueued_ += prefetch;
    }

    // Check if the prefetch buffer is not empty.
    if (!buffer_.empty()) {
      auto result = std::move(buffer_.front());
      buffer_.pop();
      return result;
    }
  }

  // If prefetch buffer is empty, read the next element from the parent.
  mutex_lock input_lock_(input_mu_);
  {
    mutex_lock state_lock(state_mu_);
    // Check if a prefetch task completed and pushed anything into the
    // buffer. Otherwise we might accidentally produce elements out of order.
    if (!buffer_.empty()) {
      auto result = std::move(buffer_.front());
      buffer_.pop();
      return result;
    }
  }

  auto next = GetNextElement(exec_ctx);
  if (internal::IsConcreteAndEmpty(next) || next.eof.IsError()) Cancel();

  return next;
}

}  // namespace io
}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_IO_H_
