/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

//===- jitrt_compiler.cc - ------------------------------------------------===//
// Reference JitRt compiler for lowering from Linalg to LLVM.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/jitrt_compiler.h"

#include <memory>
#include <utility>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/jitrt/transforms/codegen_passes.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/math/passes.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/memref/passes.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/custom_call_encoding.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"

namespace tfrt {
namespace jitrt {

using xla::runtime::CreateConvertToEntrypoint;

void RegisterDefaultJitRtDialects(mlir::DialectRegistry& registry) {
  // Register MLIR dialects supported by the compiled kernels.
  registry.insert<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                  mlir::async::AsyncDialect, mlir::cf::ControlFlowDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect, xla::runtime::RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  mlir::registerArmNeonDialectTranslation(registry);
  mlir::registerAMXDialectTranslation(registry);
  mlir::registerArmSVEDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerX86VectorDialectTranslation(registry);

  // Register other information needed for JitRt passes.
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
}

void CreateDefaultJitRtCompilationPipeline(
    mlir::OpPassManager& pm, const CompilationPipelineOptions& opts) {
  // Convert entry function to the XLA entrypoint.
  pm.addPass(CreateConvertToEntrypoint());

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Optimize operations from the math dialect before outlining compute regions
  // into functions to see all constant operands.
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::runtime::CreateMathOptimizationPass(opts.math_avx2));

  // Convert all linalg operations to parallel loops.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize generated scf.parallel operations to remove single iterations.
  pm.addPass(mlir::createCanonicalizerPass());

  // Convert scf.parallel operations into async work sharding loops.
  if (opts.num_worker_threads > 1) {
    pm.addPass(CreateCostDrivenAsyncParallelForPass(
        /*asyncDispatch=*/false, /*numWorkerThreads=*/opts.num_worker_threads,
        /*legacyBehavior=*/!opts.cost_driven_async_parallel_for));

    // Run canonicalization after async-parallel-for pass to remove async
    // operations that are not needed for executing small and cheap loops.
    pm.addPass(mlir::createCanonicalizerPass());

    // Cleanup unused async work dispatch functions after canonicalization.
    pm.addPass(mlir::createSymbolDCEPass());
  }

  // Lower from high level async operations to async runtime.
  pm.addPass(mlir::createAsyncToAsyncRuntimePass());

  // Add async.runtime reference counting operations.
  pm.addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());

  // Expand math operations into std/arith dialect operations.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandOpsPass());

  // Add alignment attribute to all memref allocations.
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::runtime::CreateAlignedAllocationsPass(opts.alignment));

  // Lower everything down to LLVM dialect.
  pm.addPass(mlir::createConvertLinalgToLLVMPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());

  // Convert runtime operations and custom calls to LLVM dialect.
  xla::runtime::ConvertRuntimeToLLvmOpts rt_opts = {
      opts.populate_type_id_names, opts.populate_type_conversions,
      opts.populate_arg_encodings,
      /*populate_ret_encodings=*/{}, opts.populate_attr_encodings};
  pm.addPass(xla::runtime::CreateConvertRuntimeToLLVMPass(std::move(rt_opts)));

  // Convert async dialect to LLVM once everything else is in the LLVM dialect.
  pm.addPass(mlir::createConvertAsyncToLLVMPass());

  {
    mlir::OpPassManager& fpm = pm.nest<mlir::func::FuncOp>();
    fpm.addPass(mlir::createConvertMathToLLVMPass());
  }

  pm.addPass(mlir::createConvertMathToLibmPass());

  mlir::LowerVectorToLLVMOptions vector_to_llvm_opts;
  if (opts.math_avx2) vector_to_llvm_opts.enableX86Vector();
  pm.addPass(mlir::createConvertVectorToLLVMPass(vector_to_llvm_opts));
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertComplexToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Prepare module for translation to LLVM.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

static void CreateJitRtCompilationPipeline(mlir::OpPassManager& pm) {
  CompilationPipelineOptions copts;
  CreateDefaultJitRtCompilationPipeline(pm, copts);
}

static mlir::PassPipelineRegistration<> jitrt_pipeline(
    "jitrt-default-pipeline", "Default JitRt compilation pipeline",
    CreateJitRtCompilationPipeline);

}  // namespace jitrt
}  // namespace tfrt
