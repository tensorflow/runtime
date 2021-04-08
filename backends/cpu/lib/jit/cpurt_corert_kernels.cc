/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

//===- cpurt_corert_kernels.cc - CpuRT <-> CoreRT kernels -----------------===//
//
// C++ kernels for the CpuRT <-> CoreRT interop.

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/cpu/jit/cpurt.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cpu {
namespace jit {

// -------------------------------------------------------------------------- //
// Compile compilation unit attribute to an executable result.
// -------------------------------------------------------------------------- //

static AsyncValueRef<CompilationResult> Compile(
    CompilationUnitAttribute kernel, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // We only support functions nested in top level compiled module.
  if (kernel.nested_symbols().size() != 1)
    return EmitErrorAsync(
        exec_ctx, "compiled kernel must be referenced by one nested symbol");

  ResourceContext* res_ctx = exec_ctx.resource_context();
  auto* compilation_cache =
      res_ctx->GetOrCreateResource<CompilationResultCache>("cpurt.cache", host);

  // TODO(ezhulenev): Compute cache key based on the content of MLIR module.
  intptr_t key = exec_ctx.location().data;

  // Return compiled kernel from the cache.
  if (auto compiled = compilation_cache->Find(key)) return compiled;

  CompilationOptions opts;
  opts.num_worker_threads = host->GetNumWorkerThreads();

  string_view entrypoint = kernel.nested_symbols()[0];
  string_view module = kernel.serialized_operation();
  Expected<CompilationResult> compiled =
      CompileKernelMlirModule(module, entrypoint, opts);

  // Failed to compile kernel source.
  if (auto err = compiled.takeError())
    return EmitErrorAsync(exec_ctx, std::move(err));

  // Update the compilation cache and return the result.
  return compilation_cache->Insert(key, std::move(*compiled));
}

// -------------------------------------------------------------------------- //
// Execute compiled CPURT kernels with CoreRT interop.
// -------------------------------------------------------------------------- //

static Error ConvertTensorHandleOperandsToMemrefDesc(
    mlir::FunctionType signature, RepeatedArguments<TensorHandle> operands,
    SmallVectorImpl<MemrefDesc>* memrefs) {
  assert(memrefs->empty() && "memrefs must be empty");
  memrefs->reserve(operands.size());

  for (unsigned i = 0; i < operands.size(); ++i) {
    auto memref_ty = signature.getInput(i).cast<mlir::MemRefType>();
    if (!memref_ty)
      return MakeStringError("expected memref operand at #", i,
                             ", got: ", signature.getInput(i));

    auto memref = ConvertTensorToMemrefDesc(
        memref_ty, operands[i].GetAsyncTensor()->get<Tensor>());
    if (auto err = memref.takeError()) return err;
    memrefs->push_back(*memref);
  }

  return Error::success();
}

static void CoreRtExecute(Argument<CompilationResult> compilation_result,
                          RepeatedArguments<TensorHandle> operands,
                          RemainingResults results,
                          const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Extract tensors from tensor handle operands to pass them as the compiled
  // kernel arguments.
  SmallVector<MemrefDesc, 4> memrefs;
  if (auto err = ConvertTensorHandleOperandsToMemrefDesc(
          compilation_result->signature(), operands, &memrefs))
    return EmitErrors(results, std::move(err), exec_ctx);

  // Allocate storage for compiled kernel results.
  SmallVector<RCReference<AsyncValue>, 4> kernel_ret;
  int num_results = compilation_result->signature().getNumResults();
  kernel_ret.reserve(num_results);
  for (int i = 0; i < num_results; ++i) kernel_ret.emplace_back();

  // Execute compiled kernel and get back raw return values that we'll need to
  // wrap into TensorHandles later on.
  ReturnValueConverter converter({host, kernel_ret});
  converter.AddConversion(ReturnMemrefAsDenseHostTensor);
  converter.AddConversion(ReturnAsyncMemrefAsDenseHostTensor);
  // We skip error handling at this point and rely on error forwarding to the
  // kernel results below.
  auto err = compilation_result->Execute(memrefs, converter, exec_ctx);
  (void)err;

  // Compiled kernel should populate all expected results.
  assert(llvm::all_of(kernel_ret, [](RCReference<AsyncValue>& ref) -> bool {
    return ref.get() != nullptr;
  }));

  // If we have unavailable kernel results we'll need to extend operands
  // lifetime.
  bool unavailable_kernel_ret = false;

  // Convert Tensors returned from compiled kernel to TensorHandles.
  for (size_t i = 0; i < results.size(); ++i) {
    const RCReference<AsyncValue>& handle = results.AllocateAt<TensorHandle>(i);
    AsyncValue* ret = kernel_ret[i].get();

    // Fast path for forwarding errors to TensorHandle results.
    if (ret->IsError()) {
      results[i] = FormRef(ret);
      continue;
    }

    // Fast path when Tensor (and tensor metadata) is available synchronously.
    if (ret->IsAvailable()) {
      Tensor& tensor = ret->get<Tensor>();
      handle->emplace<TensorHandle>(
          host->GetHostDeviceRef(), tensor.metadata(),
          AsyncValueRef<Tensor>(kernel_ret[i].CopyRef()));
      continue;
    }

    // Slow path when result Tensor is not available synchronously.
    unavailable_kernel_ret = true;
    ret->AndThen([host, handle = handle.CopyRef(),
                  ref = kernel_ret[i].CopyRef()]() mutable {
      Tensor& tensor = ref->get<Tensor>();
      handle->emplace<TensorHandle>(host->GetHostDeviceRef(), tensor.metadata(),
                                    AsyncValueRef<Tensor>(std::move(ref)));
    });
  }

  // Keep operands alive if we have unavailable results.
  if (unavailable_kernel_ret)
    RunWhenReady(kernel_ret, [o = RCArray<AsyncValue>(operands.values())] {});
}

void RegisterCpuRuntimeCoreRtKernels(KernelRegistry* registry) {
  registry->AddKernel("cpurt.corert.compile", TFRT_KERNEL(Compile));
  registry->AddKernel("cpurt.corert.execute", TFRT_KERNEL(CoreRtExecute));
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
