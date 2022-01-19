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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

void setEntryPoint(ModuleOp module, wrapper::Platform platform,
                   StringRef function_name, ArrayRef<int64_t> buffer_sizes) {
  OpBuilder builder(module.getContext());

  // Create a function.
  builder.setInsertionPoint(&module.front());
  Type entry_point_type = mlir::OpaqueType::get(
      builder.getStringAttr(GpuDialect::getDialectNamespace()), "entry_point");
  mlir::FunctionType func_type = builder.getFunctionType({}, entry_point_type);
  mlir::Location loc = module->getLoc();
  FuncOp func_op =
      builder.create<FuncOp>(loc, GetEntryPointFuncName(), func_type);
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
  Operation *get_entry_point_op = builder.createOperation(state);

  // Return entry point.
  builder.create<compiler::ReturnOp>(loc, get_entry_point_op->getResult(0));
}

}  // namespace gpu
}  // namespace tfrt
