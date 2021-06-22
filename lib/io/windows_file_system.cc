/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// This file implements the WindowsFileSystem class.

#include "windows_file_system.h"

#include <Windows.h>
#include <assert.h>

#include <limits>

#include "llvm_derived/Support/raw_ostream.h"

namespace tfrt {
namespace io {
namespace {

// This class is used to read data from a random access file.
class WindowsRandomAccessFile : public RandomAccessFile {
 public:
  explicit WindowsRandomAccessFile(HANDLE hfile, const std::string& path)
      : hfile_(hfile), path_(path) {
    assert(hfile != INVALID_HANDLE_VALUE);
  }

  ~WindowsRandomAccessFile() override;

  // This class is not copyable or movable.
  WindowsRandomAccessFile(const WindowsRandomAccessFile&) = delete;
  WindowsRandomAccessFile operator=(const WindowsRandomAccessFile&) = delete;

  llvm::Expected<size_t> Read(char* buf, size_t max_count,
                              size_t offset) const override;

 private:
  HANDLE hfile_ = INVALID_HANDLE_VALUE;
  const std::string path_;
};

WindowsRandomAccessFile::~WindowsRandomAccessFile() {
  if (hfile_ != INVALID_HANDLE_VALUE) {
    if (!::CloseHandle(hfile_)) {
      tfrt::errs() << "failed to close file " << path_
                   << " due to error: " << ::GetLastError() << "\n";
    }
  }
}

llvm::Expected<size_t> WindowsRandomAccessFile::Read(char* buf,
                                                     size_t max_count,
                                                     size_t offset) const {
  size_t actual_count = 0;
  while (actual_count < max_count) {
    size_t request_count = max_count - actual_count;
    if (request_count > std::numeric_limits<std::uint32_t>::max()) {
      request_count = std::numeric_limits<std::uint32_t>::max();
    }
    DWORD read_count = 0;
    OVERLAPPED overlapped = {};
    const size_t current_offset = offset + actual_count;
    overlapped.Offset = static_cast<DWORD>(current_offset);
    overlapped.OffsetHigh = static_cast<DWORD>(current_offset >> 32);
    if (!::ReadFile(hfile_, buf, static_cast<DWORD>(request_count), &read_count,
                    &overlapped)) {
      return MakeStringError("failed to read file ", path_,
                             " due to error: ", ::GetLastError());
    }
    if (read_count == 0) break;
    actual_count += read_count;
  }
  return actual_count;
}
}  // namespace

llvm::Error WindowsFileSystem::NewRandomAccessFile(
    const std::string& path, std::unique_ptr<RandomAccessFile>* file) {
  HANDLE hfile = ::CreateFileA(path.c_str(), GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                               OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL);
  if (hfile == INVALID_HANDLE_VALUE) {
    file->reset();
    return MakeStringError("failed to open file ", path,
                           " due to error: ", ::GetLastError());
  }
  *file = std::make_unique<WindowsRandomAccessFile>(hfile, path);
  return llvm::Error::success();
}

void RegisterFileSystem(FileSystemRegistry* registry) {
  auto file_system = std::make_unique<WindowsFileSystem>();
  // The scheme is an empty string to be backward-compatible with TF.
  registry->Register("", std::move(file_system));
}


}  // namespace io
}  // namespace tfrt
