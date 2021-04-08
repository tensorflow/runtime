// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements TFRecordDataset class which reads records from TFRecord
// files into strings.

#include "tf_record_dataset.h"

#include "tfrt/io/buffered_input_stream.h"
#include "tfrt/io/file_input_stream.h"
#include "tfrt/io/file_system.h"
#include "tfrt/support/crc32c.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/raw_coding.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// Implementation for TFRecordDataset member functions
//===----------------------------------------------------------------------===//

RCReference<Iterator> TFRecordDataset::MakeIterator(
    const IteratorContext& context) {
  return TakeRef(
      host_->Construct<TFRecordDatasetIterator>(FormRef(this), context));
}

//===----------------------------------------------------------------------===//
// Implementation for TFRecordDatasetIterator member functions
//===----------------------------------------------------------------------===//
IterationResult TFRecordDatasetIterator::GetNextElement(
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  if (auto error = MaybeInitializeStream()) {
    auto async_error = MakeErrorAsyncValueRef(host, StrCat(error));
    return IterationResult::Error(std::move(async_error), 1);
  }

  bool eof = false;
  auto result = ReadRecord(&eof);

  if (eof) {
    return IterationResult::Eof(host, 1);
  }
  if (!result) {
    // Do not decode location or emit error because the local handler might have
    // been freed.
    auto error = MakeErrorAsyncValueRef(host, StrCat(result.takeError()));
    return IterationResult::Error(std::move(error), 1);
  }

  llvm::SmallVector<RCReference<AsyncValue>, 4> values;
  values.push_back(
      MakeAvailableAsyncValueRef<std::string>(host, std::move(*result)));
  return IterationResult::Values(std::move(values), host);
}

// Logic based on tensorflow/core/io/record_reader.*
// Note: RecordReader maintains the offset. For now, we're relying on
// ifstream reading sequentially.
llvm::Expected<std::string> TFRecordDatasetIterator::ReadChecksummed(
    size_t pos, size_t n, bool* eof) {
  // The crc has size uint32.
  const size_t count = n + sizeof(uint32_t);
  *eof = false;

  std::string result;
  result.clear();
  result.resize(count);

  char* buffer = &result[0];
  auto count_or_error = stream_->Read(buffer, count);
  if (!count_or_error) return count_or_error.takeError();

  if (*count_or_error < count) {
    *eof = true;
    return MakeStringError("end of file");
  }
  const uint32_t masked_crc = DecodeFixed32(result.data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(result.data(), n)) {
    return MakeStringError("data corruption at position ", pos);
  }

  result.resize(n);
  return result;
}

llvm::Expected<std::string> TFRecordDatasetIterator::ReadRecord(bool* eof) {
  *eof = false;
  auto pos = stream_->Tell();
  if (!pos) return pos.takeError();

  // Read header.
  auto header = ReadChecksummed(*pos, sizeof(uint64_t), eof);
  if (!header) return header.takeError();
  const uint64_t length = DecodeFixed64(header->data());

  // Read body.
  auto body = ReadChecksummed(*pos, length, eof);
  if (*eof) {
    *eof = false;
    return MakeStringError("truncated record at position ", *pos);
  }
  return body;
}

llvm::Error TFRecordDatasetIterator::MaybeInitializeStream() {
  if (initialization_error_) {
    return MakeStringError(initialization_error_);
  }

  if (stream_) return llvm::Error::success();

  auto* fs_registry = ::tfrt::io::FileSystemRegistry::Default();
  auto* file_system = fs_registry->Lookup("");
  if (!file_system) {
    initialization_error_ =
        MakeStringError("No file system is found for the given scheme");
    return MakeStringError(initialization_error_);
  }

  std::unique_ptr<::tfrt::io::RandomAccessFile> file;
  auto error = file_system->NewRandomAccessFile(parent_dataset_->path_, &file);
  if (error) {
    initialization_error_ = MakeStringError(error);
    return error;
  }

  stream_ = std::make_unique<::tfrt::io::FileInputStream>(std::move(file));
  if (parent_dataset_->buffer_size_ > 0) {
    stream_ = std::make_unique<::tfrt::io::BufferedInputStream>(
        std::move(stream_), parent_dataset_->buffer_size_,
        parent_dataset_->allocator_);
  }
  return llvm::Error::success();
}

}  // namespace data
}  // namespace tfrt
