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

//===- tf_record_dataset.cc -----------------------------------------------===//
//
// This file implements TFRecordDataset class which reads records from TFRecord
// files into strings.
//
//===----------------------------------------------------------------------===//

#include "tf_record_dataset.h"

#include "tfrt/support/error_util.h"

namespace tfrt {
namespace data {

namespace {
// The following is copied from tensorflow/core/platform/raw_coding.h
static const bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

inline uint32_t DecodeFixed32(const char* ptr) {
  if (kLittleEndian) {
    // Load the raw bytes
    uint32_t result;
    memcpy(&result, ptr,
           sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64_t DecodeFixed64(const char* ptr) {
  if (kLittleEndian) {
    // Load the raw bytes
    uint64_t result;
    memcpy(&result, ptr,
           sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64_t lo = DecodeFixed32(ptr);
    uint64_t hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}
}  // namespace

//===----------------------------------------------------------------------===//
// Implementation for TFRecordDataset member functions
//===----------------------------------------------------------------------===//

RCReference<Iterator> TFRecordDataset::MakeIterator() {
  return TakeRef(host_->Construct<TFRecordDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// Implementation for TFRecordDatasetIterator member functions
//===----------------------------------------------------------------------===//
IterationResult TFRecordDatasetIterator::GetNextElement(
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  bool eof = false;
  auto result = ReadRecord(&eof);

  if (eof) {
    return IterationResult::Eof(host, 1);
  }
  if (!result) {
    // Do not decode location or emit error because the local handler might have
    // been freed.
    auto error = host->MakeErrorAsyncValueRef(StrCat(result.takeError()));
    return IterationResult::Error(std::move(error), 1);
  }

  llvm::SmallVector<RCReference<AsyncValue>, 4> values;
  values.push_back(
      host->MakeAvailableAsyncValueRef<std::string>(std::move(*result)));
  return IterationResult::Values(std::move(values), host);
}

// Logic based on tensorflow/core/io/record_reader.*
// Note: RecordReader maintains the offset. For now, we're relying on
// ifstream reading sequentially.
llvm::Expected<std::string> TFRecordDatasetIterator::ReadChecksummed(
    size_t n, bool* eof) {
  // The crc has size uint32.
  const size_t count = n + sizeof(uint32_t);
  *eof = false;

  std::string result;
  result.clear();
  result.resize(count);

  char* buffer = &result[0];
  auto count_or_error = stream_->Read(buffer, count);

  if (!count_or_error) {
    return count_or_error.takeError();
  }

  if (*count_or_error < count) {
    *eof = true;
    return MakeStringError("end of file");
  }

  // TODO(rachelim): Check the checksum.
  result.resize(n);
  return result;
}

llvm::Expected<std::string> TFRecordDatasetIterator::ReadRecord(bool* eof) {
  // Read header.
  auto header = ReadChecksummed(sizeof(uint64_t), eof);
  if (!header) {
    return header.takeError();
  }
  const uint64_t length = DecodeFixed64(header->data());

  // Read body.
  auto body = ReadChecksummed(length, eof);
  if (*eof) {
    // Successfully read the header, but got eof on the body, i.e. this is
    // a partial record. Raise an error.
    *eof = false;
    return MakeStringError("failed to read body of TFRecord");
  }
  return body;
}

}  // namespace data
}  // namespace tfrt
