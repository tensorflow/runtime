// Copyright 2021 The TensorFlow Runtime Authors
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

// BefStringEmitter class to emit strings.

#include "bef_string_emitter.h"

namespace tfrt {

size_t BefStringEmitter::EmitString(string_view str) {
  auto it = offset_map_.find(str);
  if (it != offset_map_.end()) {
    return it->second;
  }

  size_t offset = size();
  EmitBytes({reinterpret_cast<const uint8_t*>(str.data()), str.size()});
  EmitByte(0);

  auto r = offset_map_.try_emplace(str, offset);
  assert(r.second);
  (void)r;

  return offset;
}

}  // namespace tfrt
