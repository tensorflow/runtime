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

//===- tf_record_dataset.h --------------------------------------*- C++ -*-===//
//
// This file declares TFRecordDataset class which reads records from TFRecord
// files into strings.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_TF_RECORD_DATASET_H_
#define TFRT_LIB_DATA_TF_RECORD_DATASET_H_

#include <fstream>

#include "dataset.h"
#include "io.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

// TFRecordDataset reads TFRecord bytes from a file.
//
// TODO(rachelim): Consider using a custom data type to represent the
// bytes read from a TFRecord file. This will make the code more type safe
// and allow for future optimizations to use mmap instead of copying bytes
// from the file onto the heap.
class TFRecordDataset : public Dataset<std::string> {
 public:
  explicit TFRecordDataset(std::string path, HostContext* host)
      : path_(std::move(path)), host_(host), allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  TFRecordDataset(const TFRecordDataset&) = delete;
  TFRecordDataset& operator=(const TFRecordDataset&) = delete;

  RCReference<Iterator<std::string>> MakeIterator() override;

 private:
  friend class TFRecordDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<TFRecordDataset>(this, allocator_);
  }

  const std::string path_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class TFRecordDatasetIterator : public io::PrefetchingIterator<std::string> {
 public:
  explicit TFRecordDatasetIterator(RCReference<TFRecordDataset> parent_dataset)
      : io::PrefetchingIterator<std::string>(256, 64),
        parent_dataset_(std::move(parent_dataset)),
        stream_(parent_dataset_->path_.c_str(), std::ios_base::binary) {}

  // This class is not copyable or movable.
  TFRecordDatasetIterator(const TFRecordDatasetIterator&) = delete;
  TFRecordDatasetIterator& operator=(const TFRecordDatasetIterator&) = delete;

 protected:
  // Reads the next record from the input file. Returns empty AsyncValueRef if
  // input file is exhausted. Returns error async value if failed to read
  // the next record.
  IterationResult<std::string> GetNextElement(
      const ExecutionContext& exec_ctx) final;

 private:
  void Destroy() override {
    internal::DestroyImpl<TFRecordDatasetIterator>(this,
                                                   parent_dataset_->allocator_);
  }

  // Reads n + 4 bytes from the input stream and verifies that the checksum of
  // the first n bytes is stored in the last 4 bytes. Updates *eof to true
  // iff stream_ is already at eof and no bytes are read. Returns an error if
  // less than n + 4 bytes can be read, or the checksum doesn't match.
  // Otherwise, advances the input stream by n + 4 bytes and returns the first
  // n bytes.
  // If eof is set to true, caller should not process the return value.
  llvm::Expected<std::string> ReadChecksummed(size_t n, bool* eof);

  // Reads a record from the input stream and advances the input stream to point
  // to the start of the next record. Updates *eof to true iff stream_ is
  // already at the end of file and there is no error. Otherwise, returns the
  // record or an error. If eof is set to true, caller should not process the
  // return value.
  llvm::Expected<std::string> ReadRecord(bool* eof);

  RCReference<TFRecordDataset> parent_dataset_;
  std::ifstream stream_;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_TF_RECORD_DATASET_H_
