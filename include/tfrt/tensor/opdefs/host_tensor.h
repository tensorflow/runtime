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

// MLIR op definitions for host tensor dialects.

#ifndef TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_
#define TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace tfrt {
namespace ht {

class HostTensorDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "ht"; }
  explicit HostTensorDialect(MLIRContext *context);

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &os) const override;
};

/// The host buffer descriptor type.
class HostBufferType
    : public Type::TypeBase<HostBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace ht
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/host_tensor.h.inc"

#endif  // TFRT_TENSOR_OPDEFS_HOST_TENSOR_H_
