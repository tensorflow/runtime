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

//===- BEF to MLIR translation --------------------------------------------===//
//
// This file implements a mlir translation for the bef-to-mlir converter. It
// opens up an BEF file specified on the command line and converts it to a mlir
// file at specified location.
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_converter/bef_to_mlir.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {

mlir::OwningOpRef<mlir::ModuleOp> BEFToMLIRTranslate(
    llvm::SourceMgr &source_mgr, mlir::MLIRContext *context) {
  mlir::DialectRegistry registry;
  RegisterTFRTDialects(registry);
  RegisterTFRTCompiledDialects(registry);
  context->appendDialectRegistry(registry);
  for (const auto &dialect_name : context->getAvailableDialects()) {
    context->getOrLoadDialect(dialect_name);
  }

  const llvm::MemoryBuffer *input =
      source_mgr.getMemoryBuffer(source_mgr.getMainFileID());
  mlir::Location location =
      mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);

  source_mgr.setDiagHandler([](const llvm::SMDiagnostic &diag, void *) {
    llvm::SMDiagnostic bef_diag(diag.getFilename(), diag.getKind(),
                                diag.getMessage());
    bef_diag.print(nullptr, llvm::errs());
  });

  auto *buffer_start = input->getBufferStart();
  auto buffer_size = input->getBufferSize();
  llvm::ArrayRef<uint8_t> bef_file;

  // Handle BefBuffer alignment.
  BefBuffer aligned_bef_buffer;
  if (reinterpret_cast<uint64_t>(buffer_start) % GetRequiredBefAlignment()) {
    aligned_bef_buffer.resize(buffer_size);
    std::memcpy(aligned_bef_buffer.data(), buffer_start, buffer_size);
    bef_file = llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(aligned_bef_buffer.data()),
        buffer_size);
  } else {
    bef_file = llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer_start), buffer_size);
  }

  if (bef_file.empty()) {
    mlir::emitError(location) << "BEF file is empty.";
    return {};
  }

  mlir::SourceMgrDiagnosticHandler source_mgr_diag_handler(source_mgr, context);

  return ConvertBEFToMLIR(location, bef_file, context);
}

}  // namespace tfrt
