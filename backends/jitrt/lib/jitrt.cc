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

#include "tfrt/jitrt/jitrt.h"

#include "tfrt/dtype/dtype.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"

namespace tfrt {
namespace jitrt {

using xla::runtime::MemrefDesc;

Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    return MemrefDesc(dht->shape().GetRank(), dht->dtype(),
                      const_cast<void*>(dht->data()), 0,
                      [&](auto sizes, auto strides) {
                        dht->shape().GetDimensions(sizes);
                        dht->shape().GetStrides(strides);
                      });
  }

  return MakeStringError("unsupported tensor type: ", tensor.tensor_type());
}

}  // namespace jitrt
}  // namespace tfrt
