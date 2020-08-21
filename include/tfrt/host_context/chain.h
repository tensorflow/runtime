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

//===- chain.h - Control dependence representation --------------*- C++ -*-===//
//
// Chain is a control dependence between kernels. Its runtime representation is
// a zero sized value.
//
// ReadyChain is a singleton that holds an AsyncValueRef to a Chain for each
// HostContext, to avoid repeated creation of ready chains on the heap.
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_CHAIN_H_
#define TFRT_HOST_CONTEXT_CHAIN_H_

#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

struct Chain {};

class ReadyChain {
 public:
  ReadyChain(const ReadyChain&) = delete;
  ReadyChain& operator=(const ReadyChain&) = delete;

  static ReadyChain& Get() {
    // TODO(b/162096472) Use NoDestructor when available.
    static ReadyChain& kReadyChain = *new ReadyChain();
    return kReadyChain;
  }

  AsyncValueRef<Chain> GetReadyChain(HostContext* host) {
    return all_ready_chains_[HostContextPtr(host).index()].CopyRef();
  }

 private:
  friend class HostContext;

  ReadyChain() = default;

  void Construct(HostContext* host) {
    all_ready_chains_[HostContextPtr(host).index()] =
        MakeAvailableAsyncValueRef<Chain>(host);
  }

  void Destruct(HostContext* host) {
    all_ready_chains_[HostContextPtr(host).index()].reset();
  }

  // Store a ready chain for each HostContext to avoid repeated creations of
  // ready chains on the heap.
  AsyncValueRef<Chain> all_ready_chains_[HostContextPtr::kDummyIndex];
};

AsyncValueRef<Chain> GetReadyChain(HostContext* host);

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_CHAIN_H_
