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

// This file implements MLIR operations for the dense host tensor sync dialect.

#include "tfrt/tensor/opdefs/dense_host_tensor_sync.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/tensor/opdefs/tensor.h"

namespace tfrt {
namespace dht {

//===----------------------------------------------------------------------===//
// DenseHostTensor Dialect
//===----------------------------------------------------------------------===//

DenseHostTensorSyncDialect::DenseHostTensorSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_dht_sync", context,
              TypeID::get<DenseHostTensorSyncDialect>()) {
  context->getOrLoadDialect<tfrt::t::TensorDialect>();
  allowUnknownTypes();
  allowUnknownOperations();
  addOperations<
#define GET_OP_LIST
#include "tfrt/tensor/opdefs/dense_host_tensor_sync.cpp.inc"
      >();
}

}  // namespace dht
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/tensor/opdefs/dense_host_tensor_sync.cpp.inc"
