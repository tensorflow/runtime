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

// This file declares the FileInputStream class.

#ifndef TFRT_IO_FILE_INPUT_STREAM_H_
#define TFRT_IO_FILE_INPUT_STREAM_H_

#include "tfrt/io/file_system.h"
#include "tfrt/io/input_stream.h"

namespace tfrt {
namespace io {

class FileInputStream : public InputStream {
 public:
  explicit FileInputStream(std::unique_ptr<RandomAccessFile> file)
      : file_(std::move(file)) {}

  // This class is not copyable or movable.
  FileInputStream(const FileInputStream&) = delete;
  FileInputStream& operator=(const FileInputStream&) = delete;

  llvm::Expected<size_t> Read(char* buf, size_t max_count) override;

  llvm::Expected<size_t> Tell() override;

 private:
  std::unique_ptr<RandomAccessFile> file_;
  size_t offset_ = 0;
};

}  // namespace io
}  // namespace tfrt

#endif  // TFRT_IO_FILE_INPUT_STREAM_H_
