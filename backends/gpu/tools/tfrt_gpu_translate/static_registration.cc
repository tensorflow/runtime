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

//===- MLIR translation ---------------------------------------------------===//
//
// Register translation functions for gpu dialects.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef_converter/bef_to_mlir_translate.h"
#include "tfrt/bef_converter/mlir_to_bef_translate.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/init_tfrt_dialects.h"

static mlir::TranslateFromMLIRRegistration mlir_to_bef_registration(
    "mlir-to-bef", tfrt::MLIRToBEFTranslate,
    [](mlir::DialectRegistry &registry) {
      tfrt::RegisterTFRTDialects(registry);
      registry.insert<tfrt::gpu::GpuDialect>();
    });

static mlir::TranslateToMLIRRegistration bef_to_mlir_registration(
    "bef-to-mlir", [](llvm::SourceMgr &source_mgr, mlir::MLIRContext *context) {
      mlir::DialectRegistry registry;
      registry.insert<tfrt::gpu::GpuDialect>();
      context->appendDialectRegistry(registry);
      return tfrt::BEFToMLIRTranslate(source_mgr, context);
    });
