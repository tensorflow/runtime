// Copyright 2021 The TensorFlow Runtime Authors
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

#include <iterator>
#include <string>
#include <utility>

#include "../gpu_entry_point.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/gpu/kernels/gpu_ops.h"
#include "tfrt/gpu/passes/passes.h"

namespace tfrt {
namespace gpu {

namespace {

struct SetEntryPointPass
    : public PassWrapper<SetEntryPointPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetEntryPointPass)

  SetEntryPointPass() = default;
  SetEntryPointPass(const SetEntryPointPass &) {}

  Option<wrapper::Platform> platform{
      *this, "platform",
      llvm::cl::values(
          llvm::cl::OptionEnumValue{
              "CUDA", static_cast<int>(wrapper::Platform::CUDA), ""},
          llvm::cl::OptionEnumValue{
              "ROCm", static_cast<int>(wrapper::Platform::ROCm), ""})};
  Option<std::string> function_name{*this, "function_name"};
  ListOption<int64_t> buffer_sizes{*this, "buffer_sizes"};

 private:
  StringRef getArgument() const final { return "tfrt-set-entry-point"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<compiler::TFRTDialect, GpuDialect>();
  }

  void runOnOperation() override;
};

}  // namespace

static void SetEntryPoint(ModuleOp module, wrapper::Platform platform,
                          StringRef function_name,
                          ArrayRef<int64_t> buffer_sizes) {
  OpBuilder builder(module.getContext());

  // Create a function.
  builder.setInsertionPoint(&module.front());
  Type entry_point_type = mlir::OpaqueType::get(
      builder.getStringAttr(GpuDialect::getDialectNamespace()), "entry_point");
  mlir::FunctionType func_type = builder.getFunctionType({}, entry_point_type);
  mlir::Location loc = module->getLoc();
  mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(
      loc, GetEntryPointFuncName(), func_type);
  builder.setInsertionPointToEnd(func_op.addEntryBlock());

  // Create an op that returns the entry point.
  auto buffer_sizes_attr = builder.getI64ArrayAttr(buffer_sizes);
  auto function_name_attr = builder.getStringAttr(function_name);
  auto platform_attr = PlatformAttr::get(builder.getContext(), platform);
  auto version_attr = builder.getIntegerAttr(builder.getIntegerType(64),
                                             GetEntryPointVersion());
  SmallVector<NamedAttribute, 4> attributes = {
      builder.getNamedAttr("buffer_sizes", buffer_sizes_attr),
      builder.getNamedAttr("function_name", function_name_attr),
      builder.getNamedAttr("platform", platform_attr),
      builder.getNamedAttr("version", version_attr),
  };
  OperationState state(loc, GetEntryPointOpName(), {}, entry_point_type,
                       attributes);
  Operation *get_entry_point_op = builder.create(state);

  // Return entry point.
  builder.create<compiler::ReturnOp>(loc, get_entry_point_op->getResult(0));
}

void SetEntryPointPass::runOnOperation() {
  if (!platform.hasValue()) {
    getOperation()->emitError() << "Unspecified 'platform' option";
    return signalPassFailure();
  }

  func::FuncOp func_op;
  if (function_name.hasValue()) {
    func_op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        getOperation(), StringAttr::get(&getContext(), function_name));
    if (!func_op) {
      getOperation()->emitError()
          << "Function '" << function_name << "' not found";
      return signalPassFailure();
    }
  } else {
    auto funcs = getOperation().getOps<func::FuncOp>();
    if (funcs.empty() || ++funcs.begin() != funcs.end()) {
      getOperation()->emitError() << "Expected exactly one function";
      return signalPassFailure();
    }
    func_op = *funcs.begin();
  }

  SetEntryPoint(getOperation(), platform, func_op.getSymName(), buffer_sizes);
}

std::unique_ptr<OperationPass<ModuleOp>> CreateSetEntryPointPass() {
  return std::make_unique<SetEntryPointPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateSetEntryPointPass(
    wrapper::Platform platform, StringRef function_name,
    ArrayRef<int64_t> buffer_sizes) {
  auto pass = std::make_unique<SetEntryPointPass>();
  pass->platform = platform;
  pass->function_name = function_name.str();
  pass->buffer_sizes = buffer_sizes;
  return pass;
}

}  // namespace gpu
}  // namespace tfrt
