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

//===- BEF to MLIR translation registration -------------------------------===//
//
// This file registrates BEF to MLIR translation.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef_converter/bef_to_mlir_translate.h"

namespace tfrt {
namespace {

static mlir::TranslateToMLIRRegistration registration("bef-to-mlir",
                                                      BEFToMLIRTranslate);

}  // namespace
}  // namespace tfrt
