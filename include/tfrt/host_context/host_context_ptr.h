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

//===- host_context_ptr.h - Compact pointer to HostContext ------*- C++ -*-===//
//
// This file declares HostContextPtr, a compact pointer representation for
// HostContext.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_
#define TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_

#include <cassert>
#include <cstdint>

namespace tfrt {

class HostContext;

// HostContextPtr implements a compact pointer for a HostContext by storing the
// instance index of the HostContext object. It is intended to be used in places
// where saving the memory space is important, otherwise, HostContext* should be
// used.
class HostContextPtr {
 public:
  // Implicitly convert HostContext* to HostContextPtr.
  HostContextPtr(HostContext* host);  // NOLINT
  HostContextPtr() = default;

  HostContext* operator->() const { return get(); }

  HostContext& operator*() const { return *get(); }

  HostContext* get() const;

  explicit operator bool() const { return index_ == kDummyIndex; }

 private:
  friend class HostContext;
  friend class ReadyChain;

  explicit HostContextPtr(int index) : index_{static_cast<uint8_t>(index)} {
    assert(index < kDummyIndex);
  }
  uint8_t index() const { return index_; }

  static constexpr uint8_t kDummyIndex = 255;
  const uint8_t index_ = kDummyIndex;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_HOST_CONTEXT_PTR_H_
