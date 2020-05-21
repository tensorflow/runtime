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
#include "prefetch_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// PrefetchDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> PrefetchDataset::MakeIterator() {
  return TakeRef(host_->Construct<PrefetchDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// PrefetchDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult PrefetchDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  while (buffer_.size() < parent_dataset_->prefetch_num_) {
    buffer_.push(input_iterator_->GetNext(exec_ctx));
  }
  auto result = std::move(buffer_.front());
  buffer_.pop();
  return result;
}

}  // namespace data
}  // namespace tfrt
