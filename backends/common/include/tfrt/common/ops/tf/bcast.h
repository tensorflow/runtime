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

// Helper functions for Tensorflow broadcasting rules.
//
// Refer to https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html for
// the exact broadcasting behavior.

#ifndef TFRT_BACKENDS_COMMON_OPS_TF_BCAST_H_
#define TFRT_BACKENDS_COMMON_OPS_TF_BCAST_H_

#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// Argument broadcast specification defines how to get from the argument
// shape to the result shape:
//
// Example:
//
//    Tensor arg = ...
//    Tensor res = ...
//    auto bcast = GetArgumentBCast(arg.shape(), res.shape());
//
//    // Broadcast `arg` to `res` shape.
//    arg.reshape(bcast->reshape()).broadcast(bcast->broadcast);
//
class ArgumentBCast {
 public:
  ArgumentBCast(ArrayRef<Index> reshape, ArrayRef<Index> broadcast)
      : reshape_(reshape.begin(), reshape.end()),
        broadcast_(broadcast.begin(), broadcast.end()) {
    assert(reshape.size() == broadcast.size());
  }

  size_t rank() const { return reshape_.size(); }
  ArrayRef<Index> reshape() const { return reshape_; }
  ArrayRef<Index> broadcast() const { return broadcast_; }

 private:
  SmallVector<Index, 8> reshape_;
  SmallVector<Index, 8> broadcast_;
};

// Returns a broadcasted result shape if arguments are broadcastible.
Expected<TensorShape> GetBroadcastedShape(const TensorShape& arg0_shape,
                                          const TensorShape& arg1_shape);

// Returns a reshape and broadcast specifications to get from the
// `argument_shape` to the `result_shape`.
Expected<ArgumentBCast> GetArgumentBCast(const TensorShape& argument_shape,
                                         const TensorShape& result_shape);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_OPS_TF_BCAST_H_
