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

// This file implements the BufferedInputStream class.

#include "tfrt/io/buffered_input_stream.h"

namespace tfrt {
namespace io {

llvm::Expected<size_t> BufferedInputStream::Read(char* buf, size_t max_count) {
  if (max_count < 0) return MakeStringError("max_count should not be negative");

  if (!buffer_limit_) {
    auto error = buffer_limit_.takeError();
    buffer_limit_ = 0;
    return std::move(error);
  }

  size_t actual_count = 0;
  while (actual_count < max_count) {
    if (buffer_pos_ == *buffer_limit_) {
      buffer_limit_ = input_stream_->Read(buffer_, buffer_size_);
      buffer_pos_ = 0;
      if (!buffer_limit_) {
        auto error = buffer_limit_.takeError();
        buffer_limit_ = 0;
        return std::move(error);
      }
      if (*buffer_limit_ == 0) break;
    }
    size_t read_cnt =
        std::min(*buffer_limit_ - buffer_pos_, max_count - actual_count);
    std::memcpy(buf + actual_count, buffer_ + buffer_pos_, read_cnt);
    buffer_pos_ += read_cnt;
    actual_count += read_cnt;
  }
  stream_pos_ += actual_count;
  return actual_count;
}

llvm::Expected<size_t> BufferedInputStream::Tell() { return stream_pos_; }

}  // namespace io
}  // namespace tfrt
