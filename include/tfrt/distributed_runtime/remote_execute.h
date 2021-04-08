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

// Helper classes for remote execute
//
// This file defines list of helper classes that are common for remote
// execution.
#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_EXECUTE_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_EXECUTE_H_

#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/device.h"

namespace tfrt {
// A specification for remote_execute kernel calls.
struct RemoteExecuteSpec {
  RemoteExecuteSpec(llvm::SmallVectorImpl<RCReference<Device>>&& output_devices)
      : output_devices(std::move(output_devices)) {}
  // IDEA: Added a way to efficiently represent multiple identical devices.
  // List of output devices where the outputs of remote execute kernel will
  // reside.
  llvm::SmallVector<RCReference<Device>, 4> output_devices;
};

}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_EXECUTE_H_