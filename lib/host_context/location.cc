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

// This file contains Location related functions.

#include "tfrt/host_context/location.h"

#include "llvm/Support/raw_ostream.h"

namespace tfrt {

LocationHandler::~LocationHandler() {}

raw_ostream& operator<<(raw_ostream& os, const FileLineColLocation& loc) {
  return os << loc.filename << ":" << loc.line << ":" << loc.column;
}

raw_ostream& operator<<(raw_ostream& os, const OpaqueLocation& loc) {
  return os << loc.loc;
}

raw_ostream& operator<<(raw_ostream& os, const DecodedLocation& loc) {
  visit([&](auto& loc) { os << loc; }, loc);
  return os;
}

}  // namespace tfrt
