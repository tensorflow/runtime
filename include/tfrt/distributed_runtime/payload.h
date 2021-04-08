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

// Payload
//
// This file declares Payload for sending & receiving.

#ifndef THIRD_PARTY_TF_DISTRIBUTED_RUNTIME_PAYLOAD_H_
#define THIRD_PARTY_TF_DISTRIBUTED_RUNTIME_PAYLOAD_H_

#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

struct Payload {
  explicit Payload(llvm::SmallVector<RCReference<HostBuffer>, 4>&& buffers)
      : buffers(std::move(buffers)) {}

  llvm::SmallVector<RCReference<HostBuffer>, 4> buffers;
};

}  // namespace tfrt
#endif  // THIRD_PARTY_TF_DISTRIBUTED_RUNTIME_PAYLOAD_H_
