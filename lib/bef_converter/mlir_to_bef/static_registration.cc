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

//===- MLIR to BEF Translation Registration -------------------------------===//
//
// This file implements the registration for the mlir-to-bef converter in MLIR
// Translate infrastructure.  It opens up an mlir file specified on the command
// line and converts it to a bef file at specified location.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef_converter/mlir_to_bef_translate.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {
namespace {

static mlir::TranslateFromMLIRRegistration registration(
    "mlir-to-bef", MLIRToBEFTranslate, [](mlir::DialectRegistry& registry) {
      RegisterTFRTDialects(registry);
      RegisterTFRTCompiledDialects(registry);
    });

}  // namespace
}  // namespace tfrt
