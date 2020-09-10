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

//===- mlir_src_to_bef.cc ---------------------------------------*- C++ -*-===//
//
// This file implements a utility function to convert MLIR source code to BEF.
//
//===----------------------------------------------------------------------===//

#include "tfrt/bef_converter/mlir_src_to_bef.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {

static void registerMlirDialects(mlir::DialectRegistry& registry) {
  RegisterTFRTDialects(registry);
  registry.insert<mlir::StandardOpsDialect>();
}

BEFBuffer ConvertMLIRSrcToBEF(string_view mlir_src,
                              bool disable_optional_sections) {
  // Create MLIR module from the request.
  mlir::MLIRContext context;

  context.allowUnregisteredDialects();
  registerMlirDialects(context.getDialectRegistry());

  auto module = mlir::parseSourceString(mlir_src, &context);

  if (!module) return {};

  return ConvertMLIRToBEF(module.get(), disable_optional_sections);
}

}  // namespace tfrt
