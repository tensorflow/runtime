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

// This file declares the BufferedInputStream class which buffers data from
// another input stream.

#ifndef TFRT_IO_BUFFERED_INPUT_STREAM_H_
#define TFRT_IO_BUFFERED_INPUT_STREAM_H_

#include "tfrt/host_context/host_allocator.h"
#include "tfrt/io/input_stream.h"

namespace tfrt {
namespace io {

class BufferedInputStream : public InputStream {
 public:
  explicit BufferedInputStream(std::unique_ptr<InputStream> input_stream,
                               size_t buffer_size, HostAllocator* allocator)
      : input_stream_(std::move(input_stream)),
        allocator_(allocator),
        buffer_size_(buffer_size) {
    assert(buffer_size_ > 0);
    buffer_ = allocator_->Allocate<char>(buffer_size_);
  }

  ~BufferedInputStream() override {
    allocator_->Deallocate(buffer_, buffer_size_);
  }

  // This class is not copyable or movable.
  BufferedInputStream(const BufferedInputStream&) = delete;
  BufferedInputStream& operator=(const BufferedInputStream&) = delete;

  llvm::Expected<size_t> Read(char* buf, size_t max_count) override;

  llvm::Expected<size_t> Tell() override;

 private:
  std::unique_ptr<InputStream> input_stream_;
  HostAllocator* allocator_;
  // The pointer to the buffer.
  char* buffer_ = nullptr;
  // The size of buffer in bytes that is allocated and can be written.
  size_t buffer_size_ = 0;
  // The position of the next byte in the buffer to be read.
  size_t buffer_pos_ = 0;
  // The range [0, buffer_limit_) of the buffer holds valid bytes that can be
  // read. It is guaranteed that buffer_limit_ <= buffer_size_. It could contain
  // error from the last read of the input_stream_.
  llvm::Expected<size_t> buffer_limit_ = 0;
  // Current position in this stream.
  size_t stream_pos_ = 0;
};

}  // namespace io
}  // namespace tfrt

#endif  // TFRT_IO_BUFFERED_INPUT_STREAM_H_
