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

//===- skip_dataset.cc ----------------------------------------------------===//
//
// This file implements SkipDataset class which wraps around another Dataset
// instance and skips a specified number of elements from that dataset.
//
//===----------------------------------------------------------------------===//

#include "skip_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// SkipDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> SkipDataset::MakeIterator() {
  return TakeRef(host_->Construct<SkipDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// SkipDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult SkipDatasetIterator::GetNext(const ExecutionContext& exec_ctx) {
  if (!is_skip_done_) {
    is_skip_done_ = true;
    for (int i = 0, e = parent_dataset_->count_; i < e; ++i) {
      // Skip the first `count` values. The IterationResult returned to the
      // caller will provide the correct EOF information.
      input_iterator_->GetNext(exec_ctx);
    }
  }
  return input_iterator_->GetNext(exec_ctx);
}

}  // namespace data
}  // namespace tfrt
