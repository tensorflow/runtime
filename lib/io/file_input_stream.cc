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

//===- file_input_stream.cc -----------------------------------------------===//
//
// This file implements FileInputStream.
//
//===----------------------------------------------------------------------===//

#include "tfrt/io/file_input_stream.h"

namespace tfrt {
namespace io {

llvm::Expected<size_t> FileInputStream::Read(char* buf, size_t count) {
  if (stream_.eof()) return 0;
  stream_.read(buf, count);

  if (stream_.eof()) return stream_.gcount();
  if (stream_.fail()) return MakeStringError("failed to read file: ", path_);
  return stream_.gcount();
}

llvm::Expected<size_t> FileInputStream::Tell() {
  int64_t pos = stream_.tellg();
  if (pos < 0) {
    return MakeStringError("failed to tell the file offset");
  }
  return pos;
}

}  // namespace io
}  // namespace tfrt
