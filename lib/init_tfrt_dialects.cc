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

// This defines a helper to register all TFRT dialects.

#include "tfrt/init_tfrt_dialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/core_runtime/opdefs/core_runtime.h"
#include "tfrt/core_runtime/opdefs/sync/core_runtime.h"
#include "tfrt/tensor/opdefs/coo_host_tensor.h"
#include "tfrt/tensor/opdefs/dense_host_tensor.h"
#include "tfrt/tensor/opdefs/dense_host_tensor_sync.h"
#include "tfrt/tensor/opdefs/host_tensor.h"
#include "tfrt/tensor/opdefs/tensor.h"
#include "tfrt/tensor/opdefs/tensor_shape.h"
#include "tfrt/test_kernels/opdefs/test_kernels.h"

namespace tfrt {

void RegisterTFRTDialects(mlir::DialectRegistry &registry) {
  registry.insert<compiler::TFRTDialect>();
  registry.insert<corert::CoreRTDialect>();
  registry.insert<corert_sync::CoreRTSyncDialect>();
  registry.insert<ts::TensorShapeDialect>();
  registry.insert<dht::DenseHostTensorDialect>();
  registry.insert<dht::DenseHostTensorSyncDialect>();
  registry.insert<coo::CooHostTensorDialect>();
  registry.insert<test::TestDialect>();
  registry.insert<t::TensorDialect>();
  registry.insert<ht::HostTensorDialect>();
  registry.insert<mlir::func::FuncDialect>();
}

void RegisterTFRTCompiledDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::scf::SCFDialect>();
}

}  // namespace tfrt
