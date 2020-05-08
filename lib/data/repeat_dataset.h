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

//===- repeat_dataset.h -----------------------------------------*- C++ -*-===//
//
// This file declares RepeatDataset class which wraps around another Dataset
// instance and repeats it a specified number of times.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_REPEAT_DATASET_H_
#define TFRT_DATA_REPEAT_DATASET_H_

#include "dataset.h"

namespace tfrt {
namespace data {

template <typename... T>
class RepeatDatasetIterator;

// RepeatDataset wraps around another Dataset instance and repeats it for a
// specified number of times.
template <typename... T>
class RepeatDataset : public Dataset<T...> {
 public:
  explicit RepeatDataset(RCReference<Dataset<T...>> input_dataset,
                         int32_t epochs, HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        epochs_(epochs),
        host_(host),
        allocator_(host->allocator()) {
    // TODO(rachelim): Support infinite iteration.
    assert(epochs > 0);
  }

  // This class is not copyable or movable.
  RepeatDataset(const RepeatDataset&) = delete;
  RepeatDataset& operator=(const RepeatDataset&) = delete;

  RCReference<Iterator<T...>> MakeIterator() override;

 private:
  friend class RepeatDatasetIterator<T...>;

  void Destroy() override {
    internal::DestroyImpl<RepeatDataset<T...>>(this, allocator_);
  }

  RCReference<Dataset<T...>> input_dataset_;
  int32_t epochs_;
  HostContext* host_;
  HostAllocator* allocator_;
};

template <typename... T>
class RepeatDatasetIterator : public Iterator<T...> {
 public:
  explicit RepeatDatasetIterator(RCReference<RepeatDataset<T...>> dataset)
      : Iterator<T...>(),
        parent_dataset_(std::move(dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  // This class is not copyable or movable.
  RepeatDatasetIterator(const RepeatDatasetIterator&) = delete;
  RepeatDatasetIterator& operator=(const RepeatDatasetIterator&) = delete;

  // TODO(b/155918211): Handle asynchrous EOF from the input_iterator_
  IterationResult<T...> GetNext(const ExecutionContext& exec_ctx) override {
    auto value = input_iterator_->GetNext(exec_ctx);
    if (internal::IsConcreteAndEmpty(value) &&
        epoch_ + 1 < parent_dataset_->epochs_) {
      epoch_++;
      input_iterator_ = parent_dataset_->input_dataset_->MakeIterator();
      return input_iterator_->GetNext(exec_ctx);
    }
    return value;
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<RepeatDatasetIterator>(this,
                                                 parent_dataset_->allocator_);
  }

  RCReference<RepeatDataset<T...>> parent_dataset_;
  RCReference<Iterator<T...>> input_iterator_;

  // The current epoch number.
  int32_t epoch_ = 0;
};

template <typename... T>
RCReference<Iterator<T...>> RepeatDataset<T...>::MakeIterator() {
  return TakeRef(host_->Construct<RepeatDatasetIterator<T...>>(FormRef(this)));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_REPEAT_DATASET_H_
