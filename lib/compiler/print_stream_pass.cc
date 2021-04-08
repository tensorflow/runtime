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

#include "mlir/Pass/Pass.h"
#include "tfrt/compiler/stream_analysis.h"

namespace tfrt {
namespace compiler {
namespace {

class PrintStreamPass
    : public mlir::PassWrapper<PrintStreamPass,
                               mlir::OperationPass<mlir::FuncOp>> {
 public:
  void runOnOperation() override {
    auto func_op = getOperation();

    const auto& stream_analysis = getAnalysis<StreamAnalysis>();

    auto emit_stream = [&](mlir::Operation* op, mlir::Location loc) {
      const auto& stream = stream_analysis.GetStream(op);

      mlir::emitRemark(loc, "stream id: ")
          << stream.id() << ", stream cost: " << stream.cost()
          << ", parent stream: " << stream.parent_id();
    };

    emit_stream(nullptr, func_op.getLoc());
    for (auto& op : func_op.front()) {
      emit_stream(&op, op.getLoc());
    }
  }
};

// TODO(chky): Consider not using static initializers and register this pass
// explicitly in the relevant binaries.
static mlir::PassRegistration<PrintStreamPass> print_stream(
    "tfrt-print-stream", "A test pass for StreamAnalysis");

}  // namespace
}  // namespace compiler
}  // namespace tfrt
