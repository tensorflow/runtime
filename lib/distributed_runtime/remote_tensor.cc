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

// This file implements Remote Tensor.

#include "tfrt/distributed_runtime/remote_tensor.h"

namespace tfrt {

RemoteTensor::RemoteTensor(const TensorMetadata& metadata,
                           const RemoteObjectId& remote_object_id)
    : Tensor(metadata), remote_object_id_(remote_object_id) {}

void RemoteTensor::Print(raw_ostream& os) const {
  os << "RemoteTensor : " << remote_object_id_ << " dtype = " << dtype()
     << ", shape = " << shape();
}

}  // namespace tfrt
