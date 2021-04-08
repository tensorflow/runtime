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

// This file declares the interface to manage files in a file system.

#ifndef TFRT_IO_FILE_SYSTEM_H_
#define TFRT_IO_FILE_SYSTEM_H_

#include "llvm/ADT/StringMap.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace io {

// The priority of a FileSystem instance is used by FileSystemRegistry::Register
// to decide which file system to use if multiple FileSystem instances are
// registered for the same sceheme.
enum class FileSystemPriority : int { kDefault = 1, kHigh = 2 };

// An interface that declares operations to read bytes from a random access
// file.
class RandomAccessFile {
 public:
  explicit RandomAccessFile() {}

  virtual ~RandomAccessFile() {}

  // This method reads up to `max_count` bytes from the underlying IO source
  // starting at `offset`, into the buffer starting at `buf`. It could read less
  // than `max_count` bytes if there are less than `max_count` bytes available
  // starting at the given offset in the underlying IO source.
  //
  // On success, the number of bytes read is returned. If this number is smaller
  // than `max_count`, it indicates that the file has reached EOF after this
  // call.
  // On error, llvm::Error is returned.
  virtual llvm::Expected<size_t> Read(char* buf, size_t max_count,
                                      size_t offset) const = 0;
};

// An interface that declares operations to manage files in a file system.
class FileSystem {
 public:
  explicit FileSystem() {}

  virtual ~FileSystem() {}

  // Creates a read-only random access file at the given `path`.
  //
  // On success, stores a pointer to the new file in `file` and returns
  // llvm::Error::success(). Otherwise, stores NULL in `file` and returns the
  // error.
  virtual llvm::Error NewRandomAccessFile(
      const std::string& path, std::unique_ptr<RandomAccessFile>* file) = 0;

  // Returns the priority of this file system. The file system with the highest
  // priority will be used if multiple file systems have been registered for the
  // same scheme.
  virtual FileSystemPriority GetPriority() {
    return FileSystemPriority::kDefault;
  }
};

class FileSystemRegistry {
 public:
  explicit FileSystemRegistry() {}

  ~FileSystemRegistry() {}

  // This class is not copyable or movable.
  FileSystemRegistry(const FileSystemRegistry&) = delete;
  FileSystemRegistry& operator=(const FileSystemRegistry&) = delete;

  static FileSystemRegistry* Default() {
    static FileSystemRegistry* registry = new FileSystemRegistry();
    return registry;
  }

  // Registers `file_system` to be used for the given `scheme`. If there is
  // already a file system registered for the same scheme, overwrite the
  // existing registration if its priority is lower than than `file_system`.
  void Register(const std::string& scheme,
                std::unique_ptr<FileSystem> file_system);

  // Returns the file system registered for the given `scheme`.
  //
  // If no found system has been registered for the given `scheme`, it means
  // the application has not been linked against a static registration target
  // to register a file system for the given `scheme` via
  // FileSystemRegistry::Register(...).
  FileSystem* Lookup(const std::string& scheme);

 private:
  mutex mu_;
  llvm::StringMap<std::unique_ptr<FileSystem>> file_systems_
      TFRT_GUARDED_BY(mu_);
};

}  // namespace io
}  // namespace tfrt

#endif  // TFRT_IO_FILE_SYSTEM_H_
