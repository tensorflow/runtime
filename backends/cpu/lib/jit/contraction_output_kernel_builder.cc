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
using mlir::OpBuilder;
using mlir::ShapedType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

// Creates an empty function compatible with output kernel function signature.
FuncOp CreateOutputKernelFunc(MLIRContext* ctx, string_view name,
                              ArrayRef<Type> additional_args = {}) {
  auto f32_ty = FloatType::getF32(ctx);
  auto i64_ty = IntegerType::get(64, ctx);

  std::array<int64_t, 2> dynamic_dims = {ShapedType::kDynamicSize,
                                         ShapedType::kDynamicSize};
  auto memref_ty = MemRefType::get(dynamic_dims, f32_ty);

  SmallVector<Type, 4> arg_types = {memref_ty, i64_ty, i64_ty};
  arg_types.reserve(arg_types.size() + additional_args.size());
  for (Type arg_type : additional_args) arg_types.push_back(arg_type);

  auto function =
      mlir::FuncOp::create(mlir::UnknownLoc::get(ctx), name,
                           mlir::FunctionType::get(arg_types, {}, ctx));
  function.addEntryBlock();

  return function;
}

//----------------------------------------------------------------------------//
// Adds 1.0 to all elements in the output block.
//----------------------------------------------------------------------------//

class AddOne : public ContractionOutputKernelBuilder {
 public:
  mlir::FuncOp Build(mlir::MLIRContext* ctx) final;
};

mlir::FuncOp AddOne::Build(mlir::MLIRContext* ctx) {
  auto f32_ty = FloatType::getF32(ctx);

  FuncOp function = CreateOutputKernelFunc(ctx, "compute");

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
  mlir::FuncOp Build(mlir::MLIRContext* ctx) final;
};

mlir::FuncOp BiasAdd::Build(mlir::MLIRContext* ctx) {
  auto f32_ty = FloatType::getF32(ctx);
  auto index_ty = IndexType::get(ctx);
  auto bias_ty = MemRefType::get({-1}, f32_ty);

  FuncOp function = CreateOutputKernelFunc(ctx, "compute", {bias_ty});

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

}  // namespace

// TODO(ezhulenev): Add proper output kernel builder regigistration.
Expected<std::unique_ptr<ContractionOutputKernelBuilder>>
GetContractionOutputKernelBuilder(string_view name) {
  if (name == "AddOne") return std::make_unique<AddOne>();
  if (name == "BiasAdd") return std::make_unique<BiasAdd>();

  return MakeStringError("Unknown contraction output kernel: ", name);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
