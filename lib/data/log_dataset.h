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

// This file declares the LogDataset class.

#ifndef TFRT_LIB_DATA_LOG_DATASET_H_
#define TFRT_LIB_DATA_LOG_DATASET_H_

#include <queue>

#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/data/dataset.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

class LogDatasetIterator;

// LogDataset wraps around another dataset instance and forwards values from
// that dataset to its caller. It additionally logs the GetNext(...) calls to
// facilitate MLIR unit tests.
class LogDataset : public Dataset {
 public:
  explicit LogDataset(RCReference<Dataset> input_dataset, HostContext* host)
      : input_dataset_(std::move(input_dataset)), host_(host) {}

  // This class is not copyable or movable.
  LogDataset(const LogDataset&) = delete;
  LogDataset& operator=(const LogDataset&) = delete;

  RCReference<Iterator> MakeIterator(const IteratorContext& context) override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class LogDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<LogDataset>(this, host_->allocator());
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
};

class LogDatasetIterator : public Iterator {
 public:
  explicit LogDatasetIterator(RCReference<LogDataset> parent_dataset,
                              const IteratorContext& context)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(
            parent_dataset_->input_dataset_->MakeIterator(context)) {}

  // This class is not copyable or movable.
  LogDatasetIterator(const LogDatasetIterator&) = delete;
  LogDatasetIterator& operator=(const LogDatasetIterator&) = delete;

  IterationResult GetNext(const ExecutionContext& exec_ctx) override {
    tfrt::outs() << "LogDatasetIterator::GetNext called\n";
    return input_iterator_->GetNext(exec_ctx);
  }

 private:
  void Destroy() override {
    internal::DestroyImpl<LogDatasetIterator>(
        this, parent_dataset_->host_->allocator());
  }

  RCReference<LogDataset> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  std::queue<IterationResult> buffer_;
};

inline RCReference<Iterator> LogDataset::MakeIterator(
    const IteratorContext& context) {
  return TakeRef(host_->Construct<LogDatasetIterator>(FormRef(this), context));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_LOG_DATASET_H_
