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

//===- static_registration.cc ---------------------------------------------===//
//
// This file uses a static constructor to automatically register all of the
// kernels in this directory.  This can be used to simplify clients that don't
// care about selective registration of kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/opdefs/coo_host_tensor.h"
#include "tfrt/tensor/opdefs/dense_host_tensor.h"
#include "tfrt/tensor/opdefs/dense_host_tensor_sync.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"

namespace tfrt {
namespace ts {

// Static initialization for dialect registration.
static ::mlir::DialectRegistration<TensorShapeDialect> ts_registration;

}  // namespace ts

namespace dht {

// Static initialization for dialect registration.
static ::mlir::DialectRegistration<DenseHostTensorDialect> dht_registration;
static ::mlir::DialectRegistration<DenseHostTensorSyncDialect>
    dht_sync_registration;

}  // namespace dht

namespace coo {

static ::mlir::DialectRegistration<CooHostTensorDialect> coo_registration;
}

}  // namespace tfrt
