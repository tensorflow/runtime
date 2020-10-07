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

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "tfrt/core_runtime/op_attrs.h"
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
  Expected<FuncOp> Build(ModuleOp module, const OpAttrsRef& attrs, DType dtype,
                         ArrayRef<DType> additional_args) const final;
};

Expected<FuncOp> AddOne::Build(ModuleOp module, const OpAttrsRef& attrs,
                               DType dtype,
                               ArrayRef<DType> additional_args) const {
  MLIRContext* ctx = module.getContext();

  if (dtype != DType(DType::F32))
    return MakeStringError("AddOne supports only f32 dtype");

  if (!additional_args.empty())
    return MakeStringError("AddOne requires empty additional args");

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
        output_block(ivs) = intr::std_addf(output_block(ivs), one);
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
  Expected<FuncOp> Build(mlir::ModuleOp module, const OpAttrsRef& attrs,
                         DType dtype,
                         ArrayRef<DType> additional_args) const final;

  int GetNumAdditionalArgs() const final { return 1; }
};

Expected<FuncOp> BiasAdd::Build(ModuleOp module, const OpAttrsRef& attrs,
                                DType dtype,
                                ArrayRef<DType> additional_args) const {
  MLIRContext* ctx = module.getContext();

  if (dtype != DType(DType::F32))
    return MakeStringError("BiasAdd supports only f32 dtype");

  if (additional_args.size() != 1)
    return MakeStringError("BiasAdd requires one additional args");

  if (dtype != additional_args[0])
    return MakeStringError("Bias dtype must be the same as output dtype");

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
        Value i1 = ivs[1];
        Value bias = bias_vector(intr::std_addi(bias_offset, i1));
        output_block(ivs) = intr::std_addf(output_block(ivs), bias);
        return mlir::scf::ValueVector();
      });

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//
// FusedBatchNorm does the batch normalization of the output tensor (normalizes
// the output tensor by mean and variance, and applies (optionally) a scale to
// it, as well as an offset). See `tf.nn.batch_normalization` documentation.
//
// This is inference only fusion, and it does not compute batch mean or batch
// variance (see `training=False` in TF batch normalization documentation).
//----------------------------------------------------------------------------//

class FusedBatchNorm : public ContractionOutputKernelBuilder {
 public:
  Expected<FuncOp> Build(mlir::ModuleOp module, const OpAttrsRef& attrs,
                         DType dtype,
                         ArrayRef<DType> additional_args) const final;

  int GetNumAdditionalArgs() const final { return 1; }
};

Expected<FuncOp> FusedBatchNorm::Build(ModuleOp module, const OpAttrsRef& attrs,
                                       DType dtype,
                                       ArrayRef<DType> additional_args) const {
  MLIRContext* ctx = module.getContext();

  if (dtype != DType(DType::F32))
    return MakeStringError("FusedBatchNorm supports only f32 dtype");

  if (additional_args.size() != 4)
    return MakeStringError("FusedBatchNorm requires four additional args");

  bool all_f32 = llvm::all_of(additional_args, [](DType arg_dtype) -> bool {
    return arg_dtype == DType(DType::F32);
  });
  if (!all_f32)
    return MakeStringError("All additional arguments must be of f32 dtype");

  auto epsilon_value = attrs.GetOptional<float>("epsilon");
  if (!epsilon_value.hasValue())
    return MakeStringError(
        "missing epsilon attribute for the FusedBatchNorm fusion");

  auto f32_ty = FloatType::getF32(ctx);
  auto index_ty = IndexType::get(ctx);

  // Additional args: scale, offset, mean, variance
  auto additional_arg_ty = MemRefType::get({-1}, f32_ty);
  FuncOp function =
      CreateOutputKernelFunc(module, "fused_batch_norm",
                             {additional_arg_ty, additional_arg_ty,
                              additional_arg_ty, additional_arg_ty});

  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  edsc::MemRefBoundsCapture output_block_bounds(function.getArgument(0));
  intr::StdIndexedValue output_block(function.getArgument(0));
  intr::StdIndexedValue scale_vector(function.getArgument(3));
  intr::StdIndexedValue offset_vector(function.getArgument(4));
  intr::StdIndexedValue mean_vector(function.getArgument(5));
  intr::StdIndexedValue variance_vector(function.getArgument(6));

  Value d0 = output_block_bounds.ub(0);
  Value d1 = output_block_bounds.ub(1);

  Value c0 = intr::std_constant_index(0);
  Value c1 = intr::std_constant_index(1);

  // Offset in the additional args: scale, offset, mean, variance.
  Value args_offset = intr::std_index_cast(function.getArgument(2), index_ty);

  Value epsilon =
      intr::std_constant_float(llvm::APFloat(*epsilon_value), f32_ty);

  mlir::scf::buildLoopNest(
      edsc::ScopedContext::getBuilderRef(), edsc::ScopedContext::getLocation(),
      {c0, c0}, {d0, d1}, {c1, c1}, {},
      [&](OpBuilder& builder, Location loc, ValueRange ivs, ValueRange args) {
        edsc::ScopedContext loop_body(builder, loc);
        assert(args.empty() && "expected empty arguments");
        assert(ivs.size() == 2 && "expected two induction variable");
        Value i1 = ivs[1];

        Value x = output_block(ivs);
        Value scale = scale_vector(intr::std_addi(args_offset, i1));
        Value offset = scale_vector(intr::std_addi(args_offset, i1));
        Value mean = scale_vector(intr::std_addi(args_offset, i1));
        Value variance = scale_vector(intr::std_addi(args_offset, i1));

        Value x_centered = intr::std_subf(x, mean);
        Value scaling_factor = intr::std_mulf(
            intr::std_rsqrt(intr::std_addf(variance, epsilon)), scale);
        Value x_scaled = intr::std_mulf(x_centered, scaling_factor);
        Value x_shifted = intr::std_addf(x_scaled, offset);

        output_block(ivs) = x_shifted;

        return mlir::scf::ValueVector();
      });

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//
// Activation function.
//----------------------------------------------------------------------------//

class ActivationBuilder : public ContractionOutputKernelBuilder {
 public:
  explicit ActivationBuilder(string_view activation)
      : activation_(activation) {}

  Expected<FuncOp> Build(mlir::ModuleOp module, const OpAttrsRef& attrs,
                         DType dtype,
                         ArrayRef<DType> additional_args) const final;

  int GetNumAdditionalArgs() const final { return 0; }

 protected:
  virtual Expected<std::vector<Value>> BuildInitValues(
      MLIRContext* ctx, const OpAttrsRef& attrs) const = 0;

  virtual Value BuildOutputValue(MLIRContext* ctx, std::vector<Value> init,
                                 Value value) const = 0;

  string_view activation() const { return activation_; }

 private:
  string_view activation_;
};

Expected<FuncOp> ActivationBuilder::Build(
    ModuleOp module, const OpAttrsRef& attrs, DType dtype,
    ArrayRef<DType> additional_args) const {
  MLIRContext* ctx = module.getContext();

  // TODO(ezhulenev): Support more data types.
  if (dtype != DType(DType::F32))
    return MakeStringError(activation(), " supports only f32 dtype");

  if (!additional_args.empty())
    return MakeStringError(activation(), " requires empty additional args");

  FuncOp function = CreateOutputKernelFunc(module, "relu");

  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  edsc::MemRefBoundsCapture output_block_bounds(function.getArgument(0));
  intr::StdIndexedValue output_block(function.getArgument(0));

  Value d0 = output_block_bounds.ub(0);
  Value d1 = output_block_bounds.ub(1);

  Value c0 = intr::std_constant_index(0);
  Value c1 = intr::std_constant_index(1);

  auto init = BuildInitValues(ctx, attrs);
  if (auto err = init.takeError()) return std::move(err);

  mlir::scf::buildLoopNest(
      edsc::ScopedContext::getBuilderRef(), edsc::ScopedContext::getLocation(),
      {c0, c0}, {d0, d1}, {c1, c1}, {},
      [&](OpBuilder& builder, Location loc, ValueRange ivs, ValueRange args) {
        edsc::ScopedContext loop_body(builder, loc);
        assert(args.empty() && "expected empty arguments");
        assert(ivs.size() == 2 && "expected two induction variable");
        output_block(ivs) = BuildOutputValue(ctx, *init, output_block(ivs));
        return mlir::scf::ValueVector();
      });

  intr::std_ret();

  return function;
}

//----------------------------------------------------------------------------//
// Relu activation.
//----------------------------------------------------------------------------//

class Relu : public ActivationBuilder {
 public:
  Relu() : ActivationBuilder("Relu") {}

  Expected<std::vector<Value>> BuildInitValues(
      MLIRContext* ctx, const OpAttrsRef& attrs) const final {
    auto f32_ty = FloatType::getF32(ctx);
    Value zero = intr::std_constant_float(llvm::APFloat(0.0f), f32_ty);
    std::vector<Value> values = {zero};
    return values;
  }

  Value BuildOutputValue(MLIRContext* ctx, std::vector<Value> init,
                         Value value) const final {
    Value zero = init[0];
    return intr::std_select(edsc::op::sgt(value, zero), value, zero);
  }
};

//----------------------------------------------------------------------------//
// LakyRelu activation.
//----------------------------------------------------------------------------//

class LeakyRelu : public ActivationBuilder {
 public:
  LeakyRelu() : ActivationBuilder("LeakyRelu") {}

  Expected<std::vector<Value>> BuildInitValues(
      MLIRContext* ctx, const OpAttrsRef& attrs) const final {
    auto alpha_value = attrs.GetOptional<float>("alpha");
    if (!alpha_value.hasValue())
      return MakeStringError(
          "missing alpha attribute for the LeakuRely fusion");

    auto f32_ty = FloatType::getF32(ctx);
    Value zero = intr::std_constant_float(llvm::APFloat(0.0f), f32_ty);
    Value alpha = intr::std_constant_float(llvm::APFloat(*alpha_value), f32_ty);

    std::vector<Value> values = {zero, alpha};
    return values;
  }

  Value BuildOutputValue(MLIRContext* ctx, std::vector<Value> init,
                         Value value) const final {
    Value zero = init[0];
    Value alpha = init[1];
    return intr::std_select(edsc::op::sgt(value, zero), value,
                            intr::std_mulf(value, alpha));
  }
};

//----------------------------------------------------------------------------//
// Compose contraction output kernels.
//----------------------------------------------------------------------------//

class OutputKernelsComposition : public ContractionOutputKernelBuilder {
 public:
  explicit OutputKernelsComposition(ArrayRef<string_view> output_kernels)
      : output_kernels_(output_kernels) {}
  Expected<FuncOp> Build(mlir::ModuleOp module, const OpAttrsRef& attrs,
                         DType dtype,
                         ArrayRef<DType> additional_args) const final;

 private:
  ArrayRef<string_view> output_kernels_;
};

Expected<FuncOp> OutputKernelsComposition::Build(
    ModuleOp module, const OpAttrsRef& attrs, DType dtype,
    ArrayRef<DType> additional_args) const {
  // The number of default output kernel argumets:
  //   output_block, row offset, col offset.
  static constexpr int kNumDefautArgs = 3;

  // Create output kernel functions of all output kernels.
  llvm::SmallVector<FuncOp, 4> output_kernels;

  // Types of additional arguments for all output kernels.
  llvm::SmallVector<Type, 4> additional_args_types;

  int additional_args_offset = 0;

  for (string_view output_kernel : output_kernels_) {
    auto builder = GetContractionOutputKernelBuilder(output_kernel);
    if (auto err = builder.takeError()) return std::move(err);

    // Take dtypes corresponding to the output kernel.
    int num_additional_args = (*builder)->GetNumAdditionalArgs();
    ArrayRef<DType> dtypes = additional_args.drop_front(additional_args_offset)
                                 .take_front(num_additional_args);
    additional_args_offset += num_additional_args;

    auto function = (*builder)->Build(module, attrs, dtype, dtypes);
    if (auto err = function.takeError()) return std::move(err);

    output_kernels.push_back(*function);

    for (Type type : function->getType().getInputs().drop_front(kNumDefautArgs))
      additional_args_types.push_back(type);
  }

  // Create a function for the output kernel composition.
  auto function =
      CreateOutputKernelFunc(module, "compute", {additional_args_types});

  // Call output kernels one by one.
  OpBuilder builder(function.getBody());
  edsc::ScopedContext scope(builder, function.getLoc());

  auto output_block = function.getArgument(0);
  auto row_offset = function.getArgument(1);
  auto col_offset = function.getArgument(2);

  additional_args_offset = 0;

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
  if (name == "FusedBatchNorm") return std::make_unique<FusedBatchNorm>();
  if (name == "Relu") return std::make_unique<Relu>();
  if (name == "LeakyRelu") return std::make_unique<LeakyRelu>();

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
