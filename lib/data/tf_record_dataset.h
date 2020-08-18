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

#include "io.h"
#include "tfrt/data/dataset.h"
#include "tfrt/io/buffered_input_stream.h"
#include "tfrt/io/file_input_stream.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace data {

using ::tfrt::io::BufferedInputStream;
using ::tfrt::io::FileInputStream;
using ::tfrt::io::InputStream;

// TFRecordDataset reads TFRecord bytes from a file.
//
// TODO(rachelim): Consider using a custom data type to represent the
// bytes read from a TFRecord file. This will make the code more type safe
// and allow for future optimizations to use mmap instead of copying bytes
// from the file onto the heap.
class TFRecordDataset : public Dataset {
 public:
  explicit TFRecordDataset(std::string path, int64_t buffer_size,
                           int64_t max_prefetch_num, int64_t prefetch_threshold,
                           HostContext* host)
      : path_(std::move(path)),
        buffer_size_(buffer_size),
        max_prefetch_num_(max_prefetch_num),
        prefetch_threshold_(prefetch_threshold),
        host_(host),
        allocator_(host->allocator()) {
    assert(buffer_size_ >= 0);
  }

  // This class is not copyable or movable.
  TFRecordDataset(const TFRecordDataset&) = delete;
  TFRecordDataset& operator=(const TFRecordDataset&) = delete;

  RCReference<Iterator> MakeIterator() override;

 private:
  friend class TFRecordDatasetIterator;

  void Destroy() override {
    internal::DestroyImpl<TFRecordDataset>(this, allocator_);
  }

  const std::string path_;
  const int64_t buffer_size_;
  const int64_t max_prefetch_num_;
  const int64_t prefetch_threshold_;
  HostContext* host_;
  HostAllocator* allocator_;
};

class TFRecordDatasetIterator : public io::PrefetchingIterator {
 public:
  explicit TFRecordDatasetIterator(RCReference<TFRecordDataset> parent_dataset)
      : io::PrefetchingIterator(parent_dataset->max_prefetch_num_,
                                parent_dataset->prefetch_threshold_),
        parent_dataset_(std::move(parent_dataset)),
        stream_(new FileInputStream(parent_dataset_->path_.c_str())) {
    if (parent_dataset_->buffer_size_ > 0) {
      stream_ = std::make_unique<BufferedInputStream>(
          std::move(stream_), parent_dataset_->buffer_size_,
          parent_dataset_->allocator_);
    }
  }

  // This class is not copyable or movable.
  TFRecordDatasetIterator(const TFRecordDatasetIterator&) = delete;
  TFRecordDatasetIterator& operator=(const TFRecordDatasetIterator&) = delete;

 protected:
  // Reads the next record from the input file. Returns empty AsyncValueRef if
  // input file is exhausted. Returns error async value if failed to read
  // the next record.
  IterationResult GetNextElement(const ExecutionContext& exec_ctx) final;

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
  llvm::Expected<std::string> ReadChecksummed(size_t pos, size_t n, bool* eof);

  // Reads a record from the input stream and advances the input stream to point
  // to the start of the next record. Updates *eof to true iff stream_ is
  // already at the end of file and there is no error. Otherwise, returns the
  // record or an error. If eof is set to true, caller should not process the
  // return value.
  llvm::Expected<std::string> ReadRecord(bool* eof);

  RCReference<TFRecordDataset> parent_dataset_;
  std::unique_ptr<InputStream> stream_;
};

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_TF_RECORD_DATASET_H_
