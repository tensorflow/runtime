/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//===- string_util.cc - Impl of string utilities --------------------------===//
//
// This file implements various string utility functions.
//
// Based on tensorflow/core/platform/numbers.cc
#include "tfrt/support/string_util.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {

std::string HumanReadableNum(int64_t value) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  if (value < 0) {
    ss << "-";
    value = -value;
  }
  if (value < 1000) {
    ss << llvm::format("%lld", value);
  } else if (value >= static_cast<int64_t>(1e15)) {
    // Number bigger than 1E15; use that notation.
    ss << llvm::format("%0.3G", static_cast<double>(value));
  } else {
    static const char units[] = "kMBT";
    const char* unit = units;
    while (value >= static_cast<int64_t>(1000000)) {
      value /= static_cast<int64_t>(1000);
      ++unit;
    }
    ss << llvm::format("%.2f%c", value / 1000.0, *unit);
  }
  return s;
}

std::string HumanReadableNumBytes(int64_t num_bytes) {
  if (num_bytes == std::numeric_limits<int64_t>::min()) {
    // Special case for number with not representable negation.
    return "-8E";
  }

  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    char buf[8];  // Longest possible string is '-XXXXB'
    snprintf(buf, sizeof(buf), "%s%" PRId64, neg_str,
             static_cast<int64_t>(num_bytes));
    return std::string(buf);
  }

  static const char units[] = "KMGTPE";  // int64_t only goes up to E.
  const char* unit = units;
  while (num_bytes >= static_cast<int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return std::string(buf);
}

std::string HumanReadableElapsedTime(double seconds) {
  std::string human_readable;
  llvm::raw_string_ostream ss(human_readable);

  if (seconds < 0) {
    human_readable = "-";
    ss << "-";
    seconds = -seconds;
  }

  // Start with microseconds and keep going up to years. The comparisons must
  // account for rounding to prevent the format breaking the tested condition
  // and returning, e.g., "1e+03 us" instead of "1 ms".
  const double microseconds = seconds * 1.0e6;
  if (microseconds < 999.5) {
    ss << llvm::format("%0.3g us", microseconds);
    return ss.str();
  }
  double milliseconds = seconds * 1e3;
  if (milliseconds >= .995 && milliseconds < 1) {
    milliseconds = 1.0;
  }
  if (milliseconds < 999.5) {
    ss << llvm::format("%0.3g ms", milliseconds);
    return ss.str();
  }
  if (seconds < 60.0) {
    ss << llvm::format("%0.3g s", seconds);
    return ss.str();
  }
  seconds /= 60.0;
  if (seconds < 60.0) {
    ss << llvm::format("%0.3g min", seconds);
    return ss.str();
  }
  seconds /= 60.0;
  if (seconds < 24.0) {
    ss << llvm::format("%0.3g h", seconds);
    return ss.str();
  }
  seconds /= 24.0;
  if (seconds < 30.0) {
    ss << llvm::format("%0.3g days", seconds);
    return ss.str();
  }
  if (seconds < 365.2425) {
    ss << llvm::format("%0.3g months", seconds / 30.436875);
    return ss.str();
  }
  seconds /= 365.2425;
  ss << llvm::format("%0.3g years", seconds);
  return ss.str();
}

}  // namespace tfrt
