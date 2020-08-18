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

//===- stack_trace.cc -------------------------------------------*- C++ -*-===//
//
// This file implements capturing and printing stack traces.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <string>

#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

struct internal::StackTraceImpl {
  std::string string;
};

void internal::StackTraceDeleter::operator()(StackTraceImpl* ptr) const {
  delete ptr;
}

llvm::raw_ostream& internal::operator<<(llvm::raw_ostream& os,
                                        const StackTrace& stack_trace) {
  if (stack_trace) os << stack_trace->string;
  return os;
}

namespace {
class StackTraceOstream : public llvm::raw_ostream {
 public:
  explicit StackTraceOstream(int skip_count) : skip_count_(skip_count) {}

  std::string& str() {
    flush();
    return string_;
  }

 private:
  void write_impl(const char* ptr, size_t size) override {
    // Drop lines and decrement skip_count_ until it's 0.
    auto msg = llvm::StringRef(ptr, size).drop_until([&](char ch) {
      if (skip_count_ == 0) return true;
      if (ch == '\n') --skip_count_;
      return false;
    });
    string_.append(msg.begin(), msg.end());
  }

  uint64_t current_pos() const override { return string_.size(); }

  int skip_count_;
  std::string string_;
};

}  // namespace

StackTrace CreateStackTrace(int skip_count) {
#ifdef NDEBUG
  return nullptr;  // Disable stack traces in optimized builds.
#endif
  // FIXME: Once the trivial-abi unique_ptr patch is rolled out, we should
  // adjust this "skipcount" to be +1 and remove the tail-call inhibitor.
  StackTraceOstream os(skip_count + 2);
  llvm::sys::PrintStackTrace(os);

  auto ret = StackTrace(new internal::StackTraceImpl{std::move(os.str())});
  DoNotOptimize(ret.get());
  return ret;
}

}  // namespace tfrt
