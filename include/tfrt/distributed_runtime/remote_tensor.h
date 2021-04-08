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

// Remote Tensor
//
// This file defines RemoteTensor which represents Tensor in remote worker.
#ifndef TFRT_DISTRIBUTED_RUNTIME_REMOTE_TENSOR_H_
#define TFRT_DISTRIBUTED_RUNTIME_REMOTE_TENSOR_H_

#include "tfrt/distributed_runtime/remote_object.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {

class RemoteTensor : public Tensor, public TensorTraits<RemoteTensor> {
 public:
  RemoteTensor(const TensorMetadata& metadata,
               const RemoteObjectId& remote_object_id);

  void Print(raw_ostream& os) const override;

  static const char* name() { return "Remote"; }

  const RemoteObjectId& remote_object_id() { return remote_object_id_; }

 private:
  RemoteObjectId remote_object_id_;
};

}  // namespace tfrt
#endif  // TFRT_DISTRIBUTED_RUNTIME_REMOTE_TENSOR_H_
