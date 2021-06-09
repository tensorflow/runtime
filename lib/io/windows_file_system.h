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

// This file declares the WindowsFileSystem class.

#ifndef TFRT_LIB_IO_WINDOWS_FILE_SYSTEM_H_
#define TFRT_LIB_IO_WINDOWS_FILE_SYSTEM_H_

#include "tfrt/io/file_system.h"

namespace tfrt {
namespace io {

// This class is used to manage files in a Windows file system.
class WindowsFileSystem : public FileSystem {
 public:
  explicit WindowsFileSystem() {}

  // This class is not copyable or movable.
  WindowsFileSystem(const WindowsFileSystem&) = delete;
  WindowsFileSystem& operator=(const WindowsFileSystem&) = delete;

  llvm::Error NewRandomAccessFile(
      const std::string& path,
      std::unique_ptr<RandomAccessFile>* file) override;
};

}  // namespace io
}  // namespace tfrt

#endif  // TFRT_LIB_IO_WINDOWS_FILE_SYSTEM_H_
