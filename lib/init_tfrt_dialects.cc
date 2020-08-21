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

//===- init_tfrt_dialects.cc ----------------------------------------------===//
//
// This defines a helper to register all TFRT dialects.
//
//===----------------------------------------------------------------------===//

#include "tfrt/init_tfrt_dialects.h"

#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/core_runtime/opdefs/core_runtime.h"
#include "tfrt/data/opdefs/data_ops.h"
#include "tfrt/tensor/opdefs/coo_host_tensor.h"
#include "tfrt/tensor/opdefs/dense_host_tensor.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"
#include "tfrt/test_kernels/opdefs/test_kernels.h"

namespace tfrt {

void RegisterTFRTDialects(mlir::DialectRegistry &registry) {
  registry.insert<TFRTDialect>();
  registry.insert<corert::CoreRTDialect>();
  registry.insert<data::DataDialect>();
  registry.insert<ts::TensorShapeDialect>();
  registry.insert<dht::DenseHostTensorDialect>();
  registry.insert<coo::CooHostTensorDialect>();
  registry.insert<test::TestDialect>();
}

}  // namespace tfrt
