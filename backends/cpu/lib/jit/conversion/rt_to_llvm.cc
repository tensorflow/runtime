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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/cpu/jit/conversion/rt_passes.h"
#include "tfrt/cpu/jit/opdefs/rt_ops.h"

namespace tfrt {
namespace cpu {
namespace jit {
namespace {

using mlir::CallOp;
using mlir::ConstantOp;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::IntegerType;
using mlir::LLVMTypeConverter;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpConversionPattern;
using mlir::OperationPass;
using mlir::RewritePatternSet;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::TypeConverter;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::ValueRange;

namespace LLVM = mlir::LLVM;

#define GEN_PASS_CLASSES
#include "tfrt/cpu/jit/conversion/rt_gen_passes.h.inc"

//===----------------------------------------------------------------------===//
// Runtime C API declaration (see runtime.h header file).
//===----------------------------------------------------------------------===//

static constexpr const char *kGetResultStorage = "runtimeGetResultStorage";

struct RuntimeAPI {
  static LLVM::LLVMPointerType OpaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static FunctionType ResultStorageFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto i64 = IntegerType::get(ctx, 64);
    auto storage = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, i64}, {storage});
  }
};

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(ModuleOp module) {
  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  auto addDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name)) return;
    b.create<FuncOp>(name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addDecl(kGetResultStorage, RuntimeAPI::ResultStorageFunctionType(ctx));
}

// -------------------------------------------------------------------------- //

class RuntimeTypeConverter : public TypeConverter {
 public:
  RuntimeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertKernelContextType);
  }
  static llvm::Optional<Type> convertKernelContextType(KernelContextType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.set_output to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetOutputOpLowering : public OpConversionPattern<SetOutputOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetOutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto kernel_context = adaptor.ctx();
    auto index = rewriter.create<ConstantOp>(loc, adaptor.index());

    // Get a pointer to the result value storage from the runtime.
    auto result_ptr_ty = RuntimeAPI::OpaquePointerType(rewriter.getContext());
    auto result_ptr = rewriter.create<CallOp>(
        loc, kGetResultStorage, TypeRange(result_ptr_ty),
        ValueRange({kernel_context, index}));

    // Cast from i8* to the LLVM pointer type to store the result.
    auto stored_type = getTypeConverter()->convertType(op.value().getType());
    if (!stored_type)
      return rewriter.notifyMatchFailure(
          op, "failed to convert output type to LLVM type");

    auto casted_result_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(stored_type), result_ptr.getResult(0));

    // Store the output value into the result value storage.
    auto value = adaptor.value();
    rewriter.create<LLVM::StoreOp>(loc, value, casted_result_ptr.getResult());

    // Erase the original runtime operation.
    rewriter.eraseOp(op);

    return success();
  }
};

// -------------------------------------------------------------------------- //

class ConvertRuntimeToLLVMPass
    : public ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
  void runOnOperation() override;
};

void ConvertRuntimeToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  // Add declarations for the runtime API functions.
  AddRuntimeApiDeclarations(module);

  RuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower `rt.set_output` operation (it gets
  // converted to the llvm store operation into the result storage memory).
  LLVMTypeConverter llvmConverter(ctx);
  llvmConverter.addConversion(RuntimeTypeConverter::convertKernelContextType);

  // Lower from the runtime operations to the runtime API function calls.
  patterns.insert<SetOutputOpLowering>(llvmConverter, ctx);

  // Convert function signatures and call sites.
  populateFuncOpTypeConversionPattern(patterns, converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Set up conversion target to rewrite all runtime operations.
  ConversionTarget target(*ctx);
  target.addIllegalDialect<RuntimeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ConstantOp, UnrealizedConversionCastOp, CallOp>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass() {
  return std::make_unique<ConvertRuntimeToLLVMPass>();
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
