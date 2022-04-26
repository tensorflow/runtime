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

//===- Mlir-Opt utility ---------------------------------------------------===//
//
// Load MLIR and apply required passes on it.

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/passes/passes.h"
#include "tfrt/init_tfrt_dialects.h"
#include "tfrt/support/error_util.h"
#include "tfrt/test_kernels/opdefs/test_kernels.h"

namespace {

// Test pass to wrap tfrt_gpu ops in tfrt_gpu.streamify.
struct TestStreamifyConversionPass
    : public mlir::PassWrapper<TestStreamifyConversionPass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStreamifyConversionPass)

  StringRef getArgument() const final { return "test-streamify-conversion"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    tfrt::RegisterTFRTDialects(registry);
    tfrt::RegisterTFRTCompiledDialects(registry);
    registry.insert<tfrt::gpu::GpuDialect, mlir::arith::ArithmeticDialect,
                    mlir::cf::ControlFlowDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                    tfrt::compiler::TFRTDialect>();
  }

  void runOnOperation() override {
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    auto buffer_type = tfrt::gpu::BufferType::get(&getContext());
    converter.addConversion([&](BaseMemRefType) { return buffer_type; });
    converter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
    converter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });

    ConversionTarget wrap(getContext());
    wrap.addLegalDialect("wrap");

    RewritePatternSet patterns(&getContext());
    tfrt::gpu::populateStreamifyConversionPatterns(patterns, converter, wrap);

    ConversionTarget target(getContext());
    target
        .addLegalDialect<mlir::gpu::GPUDialect, tfrt::compiler::TFRTDialect>();
    target.addLegalDialect("other");
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<tfrt::gpu::StreamifyOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return none_of(op.getBody().getOps(),
                         [&](Operation &op) { return wrap.isLegal(&op); });
        });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect,
                  mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  tfrt::compiler::TFRTDialect, tfrt::gpu::GpuDialect,
                  tfrt::test::TestDialect>();
  PassRegistration<TestStreamifyConversionPass>();
  tfrt::gpu::registerPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TFRT pass driver\n", registry));
}
