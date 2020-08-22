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

//===- tensor_type_registration.h -------------------------------*- C++ -*-===//
//
// This file defines Tensor type and its registration.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_TENSOR_TYPE_REGISTRATION_H_
#define TFRT_TENSOR_TENSOR_TYPE_REGISTRATION_H_

#include "llvm/ADT/StringMap.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class TensorType {
 public:
  string_view name() const;

  int8_t id() const { return tensor_type_id_; }

  bool operator==(TensorType other) const {
    return tensor_type_id_ == other.id();
  }

  bool operator!=(TensorType other) const {
    return tensor_type_id_ != other.id();
  }

  static const TensorType kUnknownTensorType;

 private:
  friend class TensorTypeRegistry;

  // Constructor is hidden because all instances are managed by
  // TensorTypeRegistry.
  explicit TensorType(int8_t id) : tensor_type_id_(id) {}

  int8_t tensor_type_id_;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorType tensor_type);

TensorType RegisterStaticTensorType(string_view tensor_type_name);
TensorType GetStaticTensorType(string_view tensor_type_name);

}  // namespace tfrt

#endif  // TFRT_TENSOR_TENSOR_TYPE_REGISTRATION_H_
