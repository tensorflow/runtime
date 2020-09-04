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

//===- contraction_output_kernel_builder.h ----------------------*- C++ -*-===//
//
// Contraction output kernel builder builds a MLIR function for the contraction
// output kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_BUILDER_H_
#define TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_BUILDER_H_

#include "llvm/Support/Error.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace cpu {
namespace jit {

// Builds a contraction output kernel function.
class ContractionOutputKernelBuilder {
 public:
  virtual ~ContractionOutputKernelBuilder() = default;
  virtual mlir::FuncOp Build(mlir::MLIRContext* ctx) = 0;
};

Expected<std::unique_ptr<ContractionOutputKernelBuilder>>
GetContractionOutputKernelBuilder(string_view name);

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_BUILDER_H_
