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

//===- io.cc ----------------------------------------------------*- C++ -*-===//
//
// This file implements PrefetchingIterator, a helper class for building IO
// iterators.
//
//===----------------------------------------------------------------------===//

#include "io.h"

namespace tfrt {
namespace data {
namespace io {

IterationResult PrefetchingIterator::GetNext(const ExecutionContext& exec_ctx) {
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
