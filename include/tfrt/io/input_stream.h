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

//===- input_stream.h -------------------------------------------*- C++ -*-===//
//
// This file declares the interface for reading data from IO source.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_IO_INPUT_STREAM_H_
#define TFRT_IO_INPUT_STREAM_H_

#include <memory>

#include "tfrt/support/error_util.h"

namespace tfrt {
namespace io {

// An interface that declares operations to access IO source.
class InputStream {
 public:
  explicit InputStream() {}

  virtual ~InputStream() {}

  // Read() attempts to read up to count bytes from the input stream into the
  // buffer starting at buf. If the stream is at or past the EOF, no bytes are
  // read, and read() returns zero.
  //
  // On success, the number of bytes read is returned (zero indicates EOF if
  // count > 0). It is not an error if this number is smaller than count.
  // On error, llvm::Error is returned.
  virtual llvm::Expected<size_t> Read(char* buf, size_t count) = 0;

  // Returns the position of the next byte in the input stream to be read.
  // If the stream does not support the operation, or if it fails, the function
  // returns llvm::Error.
  virtual llvm::Expected<size_t> Tell() = 0;
};

}  // namespace io
}  // namespace tfrt

#endif  // TFRT_IO_INPUT_STREAM_H_
