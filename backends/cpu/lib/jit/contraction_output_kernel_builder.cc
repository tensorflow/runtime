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

//===- contraction_output_kernel_builder.cc ---------------------*- C++ -*-===//
//
// Contraction output kernel builders implementation using EDSC builders.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/contraction_output_kernel_builder.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace cpu {
namespace jit {

namespace {

namespace edsc = ::mlir::edsc;
namespace intr = ::mlir::edsc::intrinsics;

using mlir::FloatType;
using mlir::FuncOp;
using mlir::IndexType;
using mlir::IntegerType;
using mlir::Location;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::ShapedType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

// Returns a location based on __FILE__:__LINE__.
#define FILELOC(CTX) \
  mlir::FileLineColLoc::get(std::string(__FILE__), __LINE__, 0, CTX)

// Creates an empty function compatible with output kernel function signature.
FuncOp CreateOutputKernelFunc(ModuleOp module, string_view function_name,
                              ArrayRef<Type> additional_args = {}) {
  MLIRContext* ctx = module.getContext();

  auto f32_ty = FloatType::getF32(ctx);
  auto i64_ty = IntegerType::get(64, ctx);

  std::array<int64_t, 2> dynamic_dims = {ShapedType::kDynamicSize,
                                         ShapedType::kDynamicSize};
  auto memref_ty = MemRefType::get(dynamic_dims, f32_ty);

  SmallVector<Type, 4> arg_types = {memref_ty, i64_ty, i64_ty};
  arg_types.reserve(arg_types.size() + additional_args.size());
  for (Type arg_type : additional_args) arg_types.push_back(arg_type);

  // Generate a unique function name from `function_name`.
  int seq = 1;
  std::string unique_function_name = function_name.str();
  while (module.lookupSymbol<FuncOp>(unique_function_name)) {
    unique_function_name = llvm::formatv("{0}_{1}", function_name, seq++);
  }

  auto builder = OpBuilder::atBlockBegin(module.getBody());
  auto function =
      builder.create<FuncOp>(FILELOC(ctx), unique_function_name,
                             mlir::FunctionType::get(arg_types, {}, ctx));
  function.addEntryBlock();

  return function;
}

//----------------------------------------------------------------------------//
// Adds 1.0 to all elements in the output block.
//----------------------------------------------------------------------------//

class AddOne : public ContractionOutputKernelBuilder {
 public:
  Expected<FuncOp> Build(ModuleOp module) final;
};

Expected<FuncOp> AddOne::Build(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  auto f32_ty = FloatType::getF32(ctx);

  FuncOp function = CreateOutputKernelFunc(module, "add_one");

  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  edsc::MemRefBoundsCapture output_block_bounds(function.getArgument(0));
  intr::StdIndexedValue output_block(function.getArgument(0));

  Value d0 = output_block_bounds.ub(0);
  Value d1 = output_block_bounds.ub(1);

  Value c0 = intr::std_constant_index(0);
  Value c1 = intr::std_constant_index(1);
  Value one = intr::std_constant_float(llvm::APFloat(1.0f), f32_ty);

  mlir::scf::buildLoopNest(
      edsc::ScopedContext::getBuilderRef(), edsc::ScopedContext::getLocation(),
      {c0, c0}, {d0, d1}, {c1, c1}, {},
      [&](OpBuilder& builder, Location loc, ValueRange ivs, ValueRange args) {
        edsc::ScopedContext loop_body(builder, loc);
        assert(args.empty() && "expected empty arguments");
        assert(ivs.size() == 2 && "expected two induction variable");
        Value i0 = ivs[0];
        Value i1 = ivs[1];
        output_block(i0, i1) = intr::std_addf(output_block(i0, i1), one);
        return mlir::scf::ValueVector();
      });

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//
// BiasAdd adds bias vector to the inner dimension.
//----------------------------------------------------------------------------//

class BiasAdd : public ContractionOutputKernelBuilder {
 public:
  Expected<FuncOp> Build(mlir::ModuleOp module) final;
};

Expected<FuncOp> BiasAdd::Build(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  auto f32_ty = FloatType::getF32(ctx);
  auto index_ty = IndexType::get(ctx);
  auto bias_ty = MemRefType::get({-1}, f32_ty);

  FuncOp function = CreateOutputKernelFunc(module, "bias_add", {bias_ty});

  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  edsc::MemRefBoundsCapture output_block_bounds(function.getArgument(0));
  intr::StdIndexedValue output_block(function.getArgument(0));
  intr::StdIndexedValue bias_vector(function.getArgument(3));

  Value d0 = output_block_bounds.ub(0);
  Value d1 = output_block_bounds.ub(1);

  Value c0 = intr::std_constant_index(0);
  Value c1 = intr::std_constant_index(1);

  Value bias_offset = intr::std_index_cast(function.getArgument(2), index_ty);

  mlir::scf::buildLoopNest(
      edsc::ScopedContext::getBuilderRef(), edsc::ScopedContext::getLocation(),
      {c0, c0}, {d0, d1}, {c1, c1}, {},
      [&](OpBuilder& builder, Location loc, ValueRange ivs, ValueRange args) {
        edsc::ScopedContext loop_body(builder, loc);
        assert(args.empty() && "expected empty arguments");
        assert(ivs.size() == 2 && "expected two induction variable");
        Value i0 = ivs[0];
        Value i1 = ivs[1];
        Value bias = bias_vector(intr::std_addi(bias_offset, i1));
        output_block(i0, i1) = intr::std_addf(output_block(i0, i1), bias);
        return mlir::scf::ValueVector();
      });

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//
// Compose contraction output kernels.
//----------------------------------------------------------------------------//

class OutputKernelsComposition : public ContractionOutputKernelBuilder {
 public:
  explicit OutputKernelsComposition(ArrayRef<string_view> output_kernels)
      : output_kernels_(output_kernels) {}
  Expected<FuncOp> Build(mlir::ModuleOp module) final;

 private:
  ArrayRef<string_view> output_kernels_;
};

Expected<FuncOp> OutputKernelsComposition::Build(ModuleOp module) {
  // The number of default output kernel argumets:
  //   output_block, row offset, col offset.
  static constexpr int kNumDefautArgs = 3;

  // Create output kernel functions of all output kernels.
  llvm::SmallVector<FuncOp, 4> output_kernels;

  // Types of additional arguments for all output kernels.
  llvm::SmallVector<Type, 4> additional_args;

  for (string_view output_kernel : output_kernels_) {
    auto builder = GetContractionOutputKernelBuilder(output_kernel);
    if (auto err = builder.takeError()) return std::move(err);

    auto function = (*builder)->Build(module);
    if (auto err = function.takeError()) return std::move(err);

    output_kernels.push_back(*function);

    for (Type type : function->getType().getInputs().drop_front(kNumDefautArgs))
      additional_args.push_back(type);
  }

  // Create a function for the output kernel composition.
  auto function = CreateOutputKernelFunc(module, "compute", {additional_args});

  // Call output kernels one by one.
  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  auto output_block = function.getArgument(0);
  auto row_offset = function.getArgument(1);
  auto col_offset = function.getArgument(2);

  int additional_args_offset = 0;

  for (FuncOp output_kernel : output_kernels) {
    llvm::SmallVector<Value, 4> args = {output_block, row_offset, col_offset};

    // Pass additional function arguments from the outer function call.
    int num_additional_args = output_kernel.getNumArguments() - kNumDefautArgs;
    args.reserve(args.size() + num_additional_args);

    for (int i = 0; i < num_additional_args; ++i) {
      auto arg_idx = kNumDefautArgs + additional_args_offset + i;
      args.push_back(function.getArgument(arg_idx));
    }
    additional_args_offset += num_additional_args;

    intr::std_call(output_kernel.getCallableResults(),
                   builder.getSymbolRefAttr(output_kernel.getName()), args);
  }

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//

// TODO(ezhulenev): Add proper output kernel builder regigistration.
Expected<std::unique_ptr<ContractionOutputKernelBuilder>>
GetContractionOutputKernelBuilder(string_view name) {
  if (name == "AddOne") return std::make_unique<AddOne>();
  if (name == "BiasAdd") return std::make_unique<BiasAdd>();

  return MakeStringError("Unknown contraction output kernel: ", name);
}

}  // namespace

Expected<std::unique_ptr<ContractionOutputKernelBuilder>>
GetContractionOutputKernelBuilder(ArrayRef<string_view> output_kernels) {
  assert(!output_kernels.empty());

  // For a single output kernel return builder directly.
  if (output_kernels.size() == 1) {
    return GetContractionOutputKernelBuilder(output_kernels[0]);
  }

  // Otherwise compose multiple output kernels together.
  return std::make_unique<OutputKernelsComposition>(output_kernels);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
