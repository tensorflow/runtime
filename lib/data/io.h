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
class PrefetchingIterator : public Iterator {
 public:
  PrefetchingIterator(int32_t max_num_prefetch_elements,
                      int32_t prefetch_threshold)
      : Iterator(),
        max_num_prefetch_elements_(max_num_prefetch_elements),
        prefetch_threshold_(prefetch_threshold),
        prefetch_enqueued_(0),
        cancel_(false) {
    assert(prefetch_threshold >= 0);
    assert(prefetch_threshold <= max_num_prefetch_elements);
  }

  // Gets the next element from a prefetch buffer, and maybe launches an
  // asynchronous prefetch task to fill up the buffer. If the buffer is
  // empty, reads next element from the derived iterator.
  IterationResult GetNext(const ExecutionContext& exec_ctx) final;

 protected:
  // Reads the next element from the underlying IO source. Prefetching iterator
  // guarantees that all calls to this function will be properly synchronized.
  virtual IterationResult GetNextElement(const ExecutionContext& exec_cxt)
      TFRT_REQUIRES(input_mu_) = 0;

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

  std::queue<IterationResult> buffer_ TFRT_GUARDED_BY(state_mu_);

  const int32_t max_num_prefetch_elements_;
  const int32_t prefetch_threshold_;

  // Number of pending async prefetched elements.
  int32_t prefetch_enqueued_ TFRT_GUARDED_BY(state_mu_);

  // A flag to cancel all pending async prefetch tasks.
  std::atomic<bool> cancel_;
};

}  // namespace io
}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_IO_H_
