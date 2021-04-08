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

// This file declares MemoryDataset class.

#ifndef TFRT_DATA_MEMORY_DATASET_H_
#define TFRT_DATA_MEMORY_DATASET_H_

#include "tfrt/data/dataset.h"

namespace tfrt {
namespace data {

namespace internal {

// Recursive base case
template <size_t N>
static void CopyByValueHelper(
    const llvm::SmallVector<RCReference<AsyncValue>, 4>& from,
    RCArray<AsyncValue> to) {}

// Helper to recursively copy elements from `from` to `to`.
template <size_t N, typename T, typename... RemainingT>
static void CopyByValueHelper(
    const llvm::SmallVector<RCReference<AsyncValue>, 4>& from,
    RCArray<AsyncValue> to) {
  // Copy the element at Index or forward the error.
  auto Index = N - (sizeof...(RemainingT) + 1);
  auto& val = from[Index];
  if (val->IsError()) {
    to.values()[Index]->SetError(val->GetError());
  } else {
    to.values()[Index]->emplace<T>(val->get<T>());
  }

  // Recursively copy remaining elements.
  CopyByValueHelper<N, RemainingT...>(from, to.CopyRef());
}

template <typename... T>
static void CopyByValue(
    const llvm::SmallVector<RCReference<AsyncValue>, 4>& from,
    RCArray<AsyncValue> to) {
  CopyByValueHelper<sizeof...(T), T...>(from, to.CopyRef());
}

}  // namespace internal

template <typename... T>
class MemoryDatasetIterator;

// MemoryDataset wraps around another Dataset instance and repeats the dataset
// for arbitrary number of times. The data obtained from the underlying dataset
// will be put in an in-memory buffer until all data in the underlying dataset
// are enumerated. After that it just enumerates data in the in-memory buffer.
template <typename... T>
class MemoryDataset : public Dataset {
 public:
  explicit MemoryDataset(RCReference<Dataset> input_dataset, HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  MemoryDataset(const MemoryDataset&) = delete;
  MemoryDataset& operator=(const MemoryDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  friend class MemoryDatasetIterator<T...>;

  void Destroy() override {
    internal::DestroyImpl<MemoryDataset<T...>>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
};

template <typename... T>
class MemoryDatasetIterator : public Iterator {
 public:
  explicit MemoryDatasetIterator(RCReference<MemoryDataset<T...>> dataset,
                                 const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(
            parent_dataset_->input_dataset_->MakeIterator(context)) {}

  // This class is not copyable or movable.
  MemoryDatasetIterator(const MemoryDatasetIterator&) = delete;
  MemoryDatasetIterator& operator=(const MemoryDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override {
    HostContext* host = exec_ctx.host();
    if (!buffer_completed_) {
      auto input = input_iterator_->GetNext(exec_ctx);
      if (internal::IsConcreteAndEmpty(input) && buffer_.empty()) {
        // EOF and buffer empty; forward EOF to caller.
        return input;
      }
      if (!internal::IsConcreteAndEmpty(input)) {
        // Cache is not completed and not EOF, store the value in the cache.
        buffer_.push_back(std::move(input));
        return CopyByValue(buffer_.back(), host);
      }
      // buffer is not empty and the input_iterator has reached end.
      buffer_completed_ = true;
    }
    const int index = next_buffer_index_;
    next_buffer_index_ = (next_buffer_index_ + 1) % buffer_.size();
    return CopyByValue(buffer_[index], host);
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<MemoryDatasetIterator>(this,
                                                 parent_dataset_->allocator_);
  }

  IterationResult CopyByValue(const IterationResult& input, HostContext* host) {
    // Copy data by value so that the output result can be mutated.
    SmallVector<RCReference<AsyncValue>, 4> values_copy;
    values_copy.resize(sizeof...(T));
    internal::AllocateTupleResult<T...>(
        values_copy, host, std::make_index_sequence<sizeof...(T)>{});

    RunWhenReady(
        input.values, [input = input.CopyRef(),
                       values_copy = RCArray<AsyncValue>(values_copy)]() {
          internal::CopyByValue<T...>(input.values, values_copy.CopyRef());
        });

    return IterationResult::Pending(std::move(values_copy),
                                    input.eof.CopyRef());
  }

  RCReference<MemoryDataset<T...>> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  std::vector<IterationResult> buffer_;
  int next_buffer_index_ = 0;
  bool buffer_completed_ = false;
};

template <typename... T>
RCReference<Iterator> MemoryDataset<T...>::MakeIterator(
    const IteratorContext& context) {
  return TakeRef(
      host_->Construct<MemoryDatasetIterator<T...>>(FormRef(this), context));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_MEMORY_DATASET_H_
