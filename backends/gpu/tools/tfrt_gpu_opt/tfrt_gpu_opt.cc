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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/pass/pass.h"
#include "tfrt/init_tfrt_dialects.h"

namespace tfrt {
namespace gpu {

// Test pass to wrap tfrt_gpu ops in tfrt_gpu_conversion.async.execute.
struct TestGpuAsyncConversionPass
    : public mlir::PassWrapper<TestGpuAsyncConversionPass, FunctionPass> {
  StringRef getArgument() const final { return "test-gpu-async-conversion"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tfrt::gpu::conversion::GpuConversionDialect>();
  }

  void runOnFunction() override {
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    auto buffer_type = BufferType::get(&getContext());
    converter.addConversion([&](BaseMemRefType) { return buffer_type; });
    converter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      return builder.create<conversion::CastOp>(loc, type, inputs[0]);
    });

    ConversionTarget wrap(getContext());
    wrap.addLegalDialect("wrap");

    RewritePatternSet patterns(&getContext());
    populateGpuAsyncConversionPatterns(patterns, converter, wrap);

    ConversionTarget target(getContext());
    target.addLegalDialect("other", "tfrt", "tfrt_gpu_conversion");
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return none_of(op.body().getOps(),
                     [&](Operation &op) { return wrap.isLegal(&op); });
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace gpu
}  // namespace tfrt

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<tfrt::gpu::GpuDialect, mlir::StandardOpsDialect,
                  mlir::memref::MemRefDialect>();
  PassRegistration<tfrt::gpu::TestGpuAsyncConversionPass>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TFRT pass driver\n", registry, true));
}
