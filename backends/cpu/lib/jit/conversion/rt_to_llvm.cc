/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/cpu/jit/conversion/rt_passes.h"
#include "tfrt/cpu/jit/opdefs/rt_ops.h"

namespace tfrt {
namespace cpu {
namespace jit {

namespace {

using mlir::MLIRContext;
using mlir::ModuleOp;
namespace LLVM = mlir::LLVM;

#define GEN_PASS_CLASSES
#include "tfrt/cpu/jit/conversion/rt_gen_passes.h.inc"

class RuntimeTypeConverter : public mlir::TypeConverter {
 public:
  RuntimeTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion(convertKernelContextType);
  }
  static llvm::Optional<mlir::Type> convertKernelContextType(
      KernelContextType type) {
    return LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(type.getContext(), 8));
  }
};

class ConvertRuntimeToLLVMPass
    : public ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
  void runOnOperation() override;
};

void ConvertRuntimeToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  RuntimeTypeConverter converter;
  mlir::RewritePatternSet patterns(ctx);
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<RuntimeDialect>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return converter.isSignatureLegal(op.getType());
  });

  populateFuncOpTypeConversionPattern(patterns, converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateConvertRuntimeToLLVMPass() {
  return std::make_unique<ConvertRuntimeToLLVMPass>();
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
