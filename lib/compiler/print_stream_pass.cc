/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// This implements PrintStreamPass for testing StreamAnalysis.

#include <string>

#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "tfrt/compiler/stream_analysis.h"

namespace tfrt {
namespace compiler {
namespace {

class PrintStreamPass
    : public mlir::PassWrapper<PrintStreamPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintStreamPass)

  llvm::StringRef getArgument() const final { return "tfrt-print-stream"; }

  llvm::StringRef getDescription() const final {
    return "A test pass for StreamAnalysis";
  }

  void runOnOperation() override {
    auto func_op = getOperation();

    const auto& stream_analysis = getAnalysis<StreamAnalysis>();

    auto emit_stream = [&](mlir::Operation* op, mlir::Location loc) {
      const auto& stream = stream_analysis.GetStream(op);

      auto diag = mlir::emitRemark(loc, "stream id: ");
      diag << stream.id() << ", stream cost: " << stream.cost()
           << ", parent stream: " << stream.parent_id();

      const auto& child_streams = stream.GetChildStreams(op);

      if (!child_streams.empty()) {
        llvm::SmallVector<std::string> child_stream_ids;
        child_stream_ids.reserve(child_streams.size());
        for (const auto* stream : child_streams) {
          child_stream_ids.push_back(std::to_string(stream->id()));
        }

        diag << ", child streams: [" << llvm::join(child_stream_ids, ", ")
             << "]";
      }
    };

    emit_stream(nullptr, func_op.getLoc());
    for (auto& op : func_op.front()) {
      emit_stream(&op, op.getLoc());
    }
  }
};

// TODO(chky): Consider not using static initializers and register this pass
// explicitly in the relevant binaries.
static mlir::PassRegistration<PrintStreamPass> print_stream;

}  // namespace
}  // namespace compiler
}  // namespace tfrt
