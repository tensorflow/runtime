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

// This file implements the PosixFileSystem class.

#include "posix_file_system.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <limits>

#include "llvm_derived/Support/raw_ostream.h"

namespace tfrt {
namespace io {
namespace {

// This class is used to read data from a random access file.
class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(int fd, const std::string& path)
      : fd_(fd), path_(path) {}

  ~PosixRandomAccessFile() override;

  // This class is not copyable or movable.
  PosixRandomAccessFile(const PosixRandomAccessFile&) = delete;
  PosixRandomAccessFile operator=(const PosixRandomAccessFile&) = delete;

  llvm::Expected<size_t> Read(char* buf, size_t max_count,
                              size_t offset) const override;

 private:
  int fd_;
  const std::string path_;
};

PosixRandomAccessFile::~PosixRandomAccessFile() {
  if (fd_ < 0) return;
  if (close(fd_) < 0) {
    tfrt::errs() << "failed to close file " << path_
                 << " due to error: " << strerror(errno) << "\n";
  }
}

llvm::Expected<size_t> PosixRandomAccessFile::Read(char* buf, size_t max_count,
                                                   size_t offset) const {
  if (fd_ < 0) return MakeStringError("failed to read file ", path_);

  size_t actual_count = 0;
  while (actual_count < max_count) {
    // Some platforms, notably macs, throw EINVAL if pread is asked to read
    // more than fits in a 32-bit integer.
    size_t request_count = max_count - actual_count;
    if (request_count > std::numeric_limits<std::int32_t>::max())
      request_count = std::numeric_limits<std::int32_t>::max();

    ssize_t read_count =
        pread(fd_, buf + actual_count, request_count, offset + actual_count);
    if (read_count == 0) break;
    if (read_count < 0 && errno != EINTR && errno != EAGAIN)
      return MakeStringError("failed to read file ", path_,
                             " due to error: ", strerror(errno));
    if (read_count > 0) actual_count += read_count;
  }

  return actual_count;
}
}  // namespace

llvm::Error PosixFileSystem::NewRandomAccessFile(
    const std::string& path, std::unique_ptr<RandomAccessFile>* file) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    file->reset();
    return MakeStringError("failed to open file ", path,
                           " due to error: ", strerror(errno));
  }
  *file = std::make_unique<PosixRandomAccessFile>(fd, path);
  return llvm::Error::success();
}

void RegisterPosixFileSystem(FileSystemRegistry* registry) {
  auto file_system = std::make_unique<PosixFileSystem>();
  // The scheme is an empty string to be backward-compatible with TF.
  registry->Register("", std::move(file_system));
}

}  // namespace io
}  // namespace tfrt
