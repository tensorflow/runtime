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

// This file implements a utility function to convert MLIR source code to BEF.

#include "tfrt/bef_converter/mlir_src_to_bef.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {

static void registerMlirDialects(mlir::DialectRegistry& registry) {
  RegisterTFRTDialects(registry);
  registry.insert<mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithmeticDialect>();
}

static BefBuffer ConvertMLIRSrcToBEFImpl(string_view mlir_src,
                                         bool disable_optional_sections,
                                         mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  registerMlirDialects(registry);
  context->allowUnregisteredDialects();
  context->appendDialectRegistry(registry);

  auto module = mlir::parseSourceString(mlir_src, context);

  if (!module) return {};

  return ConvertMLIRToBEF(module.get(), disable_optional_sections);
}

BefBuffer ConvertMLIRSrcToBEF(string_view mlir_src,
                              bool disable_optional_sections,
                              mlir::MLIRContext* context) {
  if (context) {
    return ConvertMLIRSrcToBEFImpl(mlir_src, disable_optional_sections,
                                   context);
  } else {
    mlir::MLIRContext local_context;
    return ConvertMLIRSrcToBEFImpl(mlir_src, disable_optional_sections,
                                   &local_context);
  }
}

}  // namespace tfrt
