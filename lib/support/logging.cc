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

// This file defines the logging methods.

#include "tfrt/support/logging.h"

#include <chrono>
#include <sstream>
#include <thread>

#include "llvm/Support/Format.h"

namespace tfrt {
namespace internal {

LogStream::LogStream(const char* fname, int line, Severity severity)
    : severity_(severity) {
  uint64_t now_usec = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  static constexpr uint64_t kUsecPerSec = 1'000'000;
  time_t now_sec = static_cast<time_t>(now_usec / kUsecPerSec);
  int32_t remainder_usec = static_cast<int32_t>(now_usec % kUsecPerSec);
  static constexpr size_t kTimeBufferSize = 30;
  char time_buffer[kTimeBufferSize];
  strftime(time_buffer, kTimeBufferSize, "%Y-%m-%d %H:%M:%S",
           localtime(&now_sec));

  std::ostringstream tid_oss;
  tid_oss << std::this_thread::get_id();

  if (auto* slash = strrchr(fname, '/')) fname = slash + 1;

  *this << "IWEF"[static_cast<int>(severity)] << ' ' << time_buffer << '.'
        << llvm::format("%06d", remainder_usec) << ' ' << tid_oss.str() << ' '
        << fname << ':' << line << "] ";
}

LogStream::~LogStream() {
  *this << "\n";
  flush();
  if (severity_ == Severity::FATAL) abort();
}

void LogStream::write_impl(const char* ptr, size_t size) {
  llvm::errs().write(ptr, size);
  pos_ += size;
}

uint64_t LogStream::current_pos() const { return pos_; }

LogStreamFatal::LogStreamFatal(const char* file, int line)
    : LogStream(file, line, Severity::FATAL) {}

}  // namespace internal
}  // namespace tfrt
