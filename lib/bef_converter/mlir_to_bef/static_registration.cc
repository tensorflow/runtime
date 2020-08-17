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
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/support/aligned_buffer.h"

static llvm::cl::opt<bool> disable_optional_sections(  // NOLINT
    "disable-optional-sections",
    llvm::cl::desc("Disable optional sections for register types, attribute "
                   "types and attribute names."),
    llvm::cl::init(false));

namespace tfrt {
namespace {

mlir::LogicalResult ConvertMLIRToBEFTranslation(mlir::ModuleOp module,
                                                llvm::raw_ostream& output) {
  tfrt::AlignedBuffer<8> bef_file =
      tfrt::ConvertMLIRToBEF(module, disable_optional_sections);
  if (bef_file.empty()) return mlir::failure();

  // Success!
  output.write(reinterpret_cast<const char*>(bef_file.data()), bef_file.size());
  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration registration(
    "mlir-to-bef", ConvertMLIRToBEFTranslation);

}  // namespace
}  // namespace tfrt
