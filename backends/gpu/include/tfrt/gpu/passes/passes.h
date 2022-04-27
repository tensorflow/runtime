/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// MLIR pass definitions for gpu_ops library

#ifndef TFRT_GPU_PASSES_PASSES_H_
#define TFRT_GPU_PASSES_PASSES_H_

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace tfrt {
namespace gpu {
namespace wrapper {
enum class Platform;
}

namespace internal {
mlir::Value StreamifyOpConversionGetStream(mlir::Operation* parent);
mlir::Value StreamifyOpConversionGetChain(mlir::Operation* parent);
void StreamifyOpConversionSetChain(mlir::Value chain,
                                   mlir::PatternRewriter& rewriter);
}  // namespace internal

// Base class for lowering ops inside a tfrt_gpu.streamify op.
template <typename OpTy>
struct StreamifyOpConversionPattern : mlir::OpConversionPattern<OpTy> {
  using typename mlir::OpConversionPattern<OpTy>::OpAdaptor;
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    auto* parent = op->getParentOp();
    auto in_chain = internal::StreamifyOpConversionGetChain(parent);
    auto stream = internal::StreamifyOpConversionGetStream(parent);
    if (!in_chain || !stream)
      return rewriter.notifyMatchFailure(op, "Failed to get chain and stream.");
    auto out_chain = matchAndRewriteOp(op, adaptor, in_chain, stream, rewriter);
    if (failed(out_chain)) return mlir::failure();
    internal::StreamifyOpConversionSetChain(*out_chain, rewriter);
    return mlir::success();
  }

  // Lowers 'op' to schedule work on 'stream' and returns chain, or none in case
  // the rewrite failed.
  virtual mlir::FailureOr<mlir::Value> matchAndRewriteOp(
      OpTy op, OpAdaptor adaptor, mlir::Value in_chain, mlir::Value stream,
      mlir::ConversionPatternRewriter& rewriter) const = 0;
};

// Returns the size in bytes of a Integer/Float/Complex/ShapedType.
// Element sizes are rounded up to full bytes (e.g. 'memref<8xi1>' is 8 bytes).
unsigned GetTypeSizeBytes(const mlir::Type& type);

// Returns the name of the device code attribute of gpu.module ops.
mlir::StringRef GetGpuBinaryAttrName();
// Returns the name of the device constants attribute of gpu.module ops.
mlir::StringRef GetGpuConstantsAttrName();
// Returns the name of the gpu.module symbol attribute of memref.get_global ops.
mlir::StringRef GetGpuModuleAttrName();

// Returns a type converter which maps memref to !tfrt_gpu.buffer and provides
// the corresponding unrealized_conversion_cast materializers.
mlir::TypeConverter CreateMemrefToTfrtGpuConverter();

// Creates a pass which wraps ops into a tfrt_gpu.streamify op.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateStreamifyOpsPass(
    mlir::ArrayRef<std::string> op_names);

// Creates a pass which wraps the template argument ops into a
// tfrt_gpu.streamify op.
template <typename... OpTs>
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateStreamifyOpsPass() {
  std::string op_names[] = {
      static_cast<std::string>(OpTs::getOperationName())...};
  return CreateStreamifyOpsPass(op_names);
}

// Adds rewrite patterns which convert memref op to tfrt_gpu.
void PopulateMemrefConversionPatterns(mlir::RewritePatternSet& patterns,
                                      mlir::TypeConverter& converter);

// Adds passes to convert from MLIR's gpu and async dialects to TFRT. Adds
// !tfrt.chain result and !tfrt.chain, !tfrt_gpu.stream arguments to functions.
void PopulateGpuToTfrtGpuPasses(mlir::OpPassManager& pm);

// Registers all tfrt gpu passes.
void RegisterPasses();

// Creates a pass which adds a function returning the entry point information
// for the gpu executor.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSetEntryPointPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSetEntryPointPass(
    wrapper::Platform platform, mlir::StringRef function_name,
    mlir::ArrayRef<int64_t> buffer_sizes);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_PASSES_PASSES_H_
