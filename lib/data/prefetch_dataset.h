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

//===- prefetch_dataset.h ---------------------------------------*- C++ -*-===//
//
// This file declares PrefetchDataset class which wraps around another dataset
// instance and prefetches elements from the underlying dataset in an internal
// buffer.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_PREFETCH_DATASET_H_
#define TFRT_LIB_DATA_PREFETCH_DATASET_H_

#include <queue>

#include "dataset.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

template <typename... T>
class PrefetchDatasetIterator;

// PrefetchDataset class which wraps around another dataset instance and
// prefetches elements from the underlying dataset in an internal buffer.
template <typename... T>
class PrefetchDataset : public Dataset<T...> {
 public:
  explicit PrefetchDataset(RCReference<Dataset<T...>> input_dataset,
                           int32_t prefetch_num, HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        prefetch_num_(prefetch_num),
        host_(host) {}

  // This class is not copyable or movable.
  PrefetchDataset(const PrefetchDataset&) = delete;
  PrefetchDataset& operator=(const PrefetchDataset&) = delete;

  RCReference<Iterator<T...>> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class PrefetchDatasetIterator<T...>;

  void Destroy() override {
    internal::DestroyImpl<PrefetchDataset<T...>>(this, host_->allocator());
  }

  RCReference<Dataset<T...>> input_dataset_;
  int32_t prefetch_num_;
  HostContext* host_;
};

template <typename... T>
class PrefetchDatasetIterator : public Iterator<T...> {
 public:
  explicit PrefetchDatasetIterator(
      RCReference<PrefetchDataset<T...>> parent_dataset)
      : Iterator<T...>(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  // This class is not copyable or movable.
  PrefetchDatasetIterator(const PrefetchDatasetIterator&) = delete;
  PrefetchDatasetIterator& operator=(const PrefetchDatasetIterator&) = delete;

  IterationResult<T...> GetNext(const ExecutionContext& exec_ctx) override {
    while (buffer_.size() < parent_dataset_->prefetch_num_) {
      buffer_.push(input_iterator_->GetNext(exec_ctx));
    }
    auto result = std::move(buffer_.front());
    buffer_.pop();
    return result;
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<PrefetchDatasetIterator>(
        this, parent_dataset_->host_->allocator());
  }

  RCReference<PrefetchDataset<T...>> parent_dataset_;
  RCReference<Iterator<T...>> input_iterator_;
  std::queue<IterationResult<T...>> buffer_;
};

template <typename... T>
RCReference<Iterator<T...>> PrefetchDataset<T...>::MakeIterator() {
  return TakeRef(
      host_->Construct<PrefetchDatasetIterator<T...>>(FormRef(this)));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_PREFETCH_DATASET_H_
