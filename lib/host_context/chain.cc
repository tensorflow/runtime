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

// Control dependence representation
//
// This file implements GetReadyChain(), a free function version of the
// ReadyChain method to get an AsyncValueRef to a Chain for the given
// host.
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/chain.h"

namespace tfrt {

AsyncValueRef<Chain> GetReadyChain(HostContext* host) {
  return ReadyChain::Get().GetReadyChain(host);
}

template <>
AsyncValueRef<Chain> MakeAvailableAsyncValueRef<Chain>(HostContext* host) {
  return GetReadyChain(host);
}

}  // namespace tfrt
