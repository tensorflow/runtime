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

// This file declares PrefetchDataset class which wraps around another dataset
// instance and prefetches elements from the underlying dataset in an internal
// buffer.
#include "prefetch_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// PrefetchDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> PrefetchDataset::MakeIterator(
    const IteratorContext& context) {
  if (is_deterministic_)
    return TakeRef(
        host_->Construct<PrefetchDatasetIterator>(FormRef(this), context));
  return TakeRef(host_->Construct<NonDeterministicPrefetchDatasetIterator>(
      FormRef(this), context));
}

//===----------------------------------------------------------------------===//
// PrefetchDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult PrefetchDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  while (buffer_.size() < parent_dataset_->prefetch_num_ + 1) {
    buffer_.push(input_iterator_->GetNext(exec_ctx));
  }
  auto result = std::move(buffer_.front());
  buffer_.pop();
  return result;
}

//===----------------------------------------------------------------------===//
// NonDeterministicPrefetchDatasetIterator methods
//===----------------------------------------------------------------------===//

static bool AvailableAndNotEof(const IterationResult& result) {
  if (result.eof.IsConcrete() && result.eof.get()) return false;
  if (!result.eof.IsAvailable()) return false;
  for (auto& value : result.values) {
    if (!value->IsAvailable()) return false;
  }
  return true;
}

IterationResult NonDeterministicPrefetchDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  while (buffer_.size() < parent_dataset_->prefetch_num_ + 1) {
    buffer_.push_back(input_iterator_->GetNext(exec_ctx));
  }
  for (auto it = buffer_.begin(), e = buffer_.end(); it != e; ++it) {
    if (AvailableAndNotEof(*it)) {
      auto value = std::move(*it);
      buffer_.erase(it);
      return value;
    }
  }

  auto result = std::move(buffer_.front());
  buffer_.pop_front();
  return result;
}

}  // namespace data
}  // namespace tfrt
