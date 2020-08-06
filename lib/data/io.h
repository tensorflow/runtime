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

#include "tfrt/data/dataset.h"
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
  explicit PrefetchingIterator(int32_t num_worker_threads)
      : Iterator(),
        max_prefetch_num_(num_worker_threads * 8),
        prefetch_threshold_(num_worker_threads * 2),
        token_owned_(false),
        reached_eof_(false) {
    assert(num_worker_threads > 0);
  }

  // Gets the next element from a prefetch buffer, and maybe launches an
  // asynchronous prefetch task to fill up the buffer. If the buffer is
  // empty, reads next element from the derived iterator.
  IterationResult GetNext(const ExecutionContext& exec_ctx) final;

 protected:
  // Reads the next element from the underlying IO source. Prefetching iterator
  // guarantees that all calls to this function will be properly synchronized.
  //
  // GetNextElement() should avoid decoding error location. This is because the
  // location handler might have already been freed when this method is called
  // by a blocking thread. PrefetchingIterator should set error location and
  // emit the error when it forwards the error to an output value.
  virtual IterationResult GetNextElement(const ExecutionContext& exec_cxt) = 0;

 private:
  // Read data from IO source if there is more data to fetch. And it forwards
  // values from the prefetch_buffer_ to those values in the output_buffer_.
  void ReadIOSource(const ExecutionContext& exec_ctx) TFRT_EXCLUDES(mu_);

  // Forward eof and values from the given input to the given output.
  void ForwardInputToOutput(IterationResult input, IterationResult output,
                            const ExecutionContext& exec_ctx);

  // Forward values from the prefetch_buffer_ to those values in the
  // output_buffer_.
  void MaterializeOutputs(const ExecutionContext& exec_ctx);

  int OutputBufferSize() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    return output_buffer_.size();
  }

  llvm::Optional<IterationResult> DequeueOutputBuffer() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    if (output_buffer_.empty()) return llvm::None;
    auto value = std::move(output_buffer_.front());
    output_buffer_.pop();
    return value;
  }

  mutex mu_;
  // A queue of IterationResult returned by GetNextElement(...). This queue
  // does not need to be explicitly protected by lock because the implementation
  // ensures that only token owner can access it.
  std::queue<IterationResult> prefetch_buffer_;
  // A queue of IterationResult that have already been returned to the
  // GetNext(...) caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);

  // Maximum number of values to prefetch from the underlying IO source
  // in addition to meeting the number of output values already requested in the
  // output_buffer_. The total number of values in the queues of the open
  // iterators is upper bounded by max_prefetch_num_ + output_buffer_size.
  const size_t max_prefetch_num_;
  // Schedule background blocking thread to prefetch from the underlying IO
  // source if the number of prefetched values dropped below this threadhold.
  const size_t prefetch_threshold_;

  // This is a unique logical token for this iterator instance. It effectively
  // acts as a lock to ensure in-order delivery of results by guaranteeing that
  // at most one thread can take the next value from the input_iterator_ and
  // update values in prefetch_buffer_.
  bool token_owned_ TFRT_GUARDED_BY(mu_);
  // Whether the iterator has reached eof. If there exits a thread that owns the
  // token, then only that thread should access this flag.
  bool reached_eof_;
};

}  // namespace io
}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_IO_H_
