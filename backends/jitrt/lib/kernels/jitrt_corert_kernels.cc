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

//===- jitrt_corert_kernels.cc - JitRT <-> CoreRT kernels -----------------===//
//
// C++ kernels for the JitRT <-> CoreRT interop.

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/tensor_handle.h"
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
#include "tfrt/host_context/shared_context.h"
#include "tfrt/jitrt/jitrt.h"
#include "tfrt/jitrt/jitrt_compiler.h"
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
namespace jitrt {

// -------------------------------------------------------------------------- //
// Compile compilation unit attribute to an executable result.
// -------------------------------------------------------------------------- //

static AsyncValueRef<JitExecutable> Compile(CompilationUnitAttribute kernel,
                                            const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // We only support functions nested in top level compiled module.
  if (kernel.nested_symbols().size() != 1)
    return EmitErrorAsync(
        exec_ctx, "compiled kernel must be referenced by one nested symbol");

  ResourceContext* res_ctx = exec_ctx.resource_context();
  auto* jit_executable_cache =
      res_ctx->GetOrCreateResource<JitExecutableCache>("jitrt.cache");

  // TODO(ezhulenev): Compute cache key based on the content of MLIR module, or
  // better keep module fingerprint in the BEF file.
  intptr_t key = exec_ctx.location().data;

  // Maybe return JitExecutable from the cache.
  if (auto cached = jit_executable_cache->Find(key)) return cached.CopyRef();

  // Allocate a placeholder for the compiled JitExecutable.
  JitExecutableCache::Entry entry = jit_executable_cache->Allocate(key);

  // We lost the race; some other invocation will do the compilation.
  if (!entry.allocated) return entry.ptr.CopyRef();

  // Compile kernel asynchronously in the host context thread pool.
  EnqueueWork(exec_ctx, [kernel, host, ref = entry.ptr.CopyRef()]() {
    CompilationPipelineOptions copts;
    copts.num_worker_threads = host->GetNumWorkerThreads();

    CompilationOptions opts;
    opts.register_dialects = RegisterDefaultJitRtDialects;
    opts.create_compilation_pipeline = [copts](mlir::PassManager& pm) {
      CreateDefaultJitRtCompilationPipeline(pm, copts);
    };

    string_view entrypoint = kernel.nested_symbols()[0];
    string_view module = kernel.serialized_operation();

    // Instantiate new JitExecutable from the MLIR source.
    Expected<JitExecutable> jit_executable =
        JitExecutable::Instantiate(module, entrypoint, opts);

    // Set the allocated async value state to error or concrete.
    if (auto err = jit_executable.takeError())
      ref.SetError(std::move(err));
    else
      ref.emplace(std::move(*jit_executable));
  });

  return entry.ptr.CopyRef();
}

// -------------------------------------------------------------------------- //
// Execute compiled JitRT kernels with CoreRT interop.
// -------------------------------------------------------------------------- //

namespace {
// We do not record any operands information for results conversion.
struct ConversionCtx {};

// Use HostContextAsyncTaskRunner to execute all async tasks.
struct AsyncTaskRunnerContext : public SharedContext {
  explicit AsyncTaskRunnerContext(HostContext* host) : runner(host) {}
  HostContextAsyncTaskRunner runner;
};
}  // namespace

static Error ConvertTensorHandleOperandsToMemrefDesc(
    RepeatedArguments<TensorHandle> operands,
    llvm::SmallVectorImpl<MemrefDesc>* memrefs) {
  assert(memrefs->empty() && "memrefs must be empty");
  memrefs->reserve(operands.size());

  for (unsigned i = 0; i < operands.size(); ++i) {
    const Tensor& tensor = operands[i].GetAsyncTensor()->get<Tensor>();
    Expected<MemrefDesc> memref = ConvertTensorToMemrefDesc(tensor);
    if (auto err = memref.takeError()) return err;
    memrefs->push_back(std::move(*memref));
  }

  return Error::success();
}

static void ExecuteImpl(const Executable& executable,
                        const llvm::SmallVectorImpl<MemrefDesc>& memrefs,
                        RepeatedArguments<TensorHandle> operands,
                        RemainingResults results,
                        const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Allocate storage for compiled kernel results.
  llvm::SmallVector<RCReference<AsyncValue>, 4> kernel_ret;
  kernel_ret.resize(executable.num_results());

  // Execute compiled kernel and get back raw return values that we'll need to
  // wrap into TensorHandles later on.
  ConversionCtx conversion_ctx;
  ReturnValueConverter<ConversionCtx> converter{RemainingResults(kernel_ret),
                                                conversion_ctx};
  converter.AddConversion(ReturnMemrefAsDenseHostTensor<ConversionCtx>);
  converter.AddConversion(ReturnAsyncMemrefAsDenseHostTensor<ConversionCtx>);

  // Use HostContext to execute all async tasks.
  auto& runner_ctx = host->GetOrCreateSharedContext<AsyncTaskRunnerContext>();

  Executable::ExecuteOpts opts;
  opts.async_task_runner = &runner_ctx.runner;

  // We skip error handling at this point and rely on error forwarding to the
  // kernel results below.
  auto err = executable.Execute(memrefs, converter, opts);
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
      handle->emplace<TensorHandle>(host->GetHostDeviceRef(), tensor.metadata(),
                                    AsyncValueRef<Tensor>(kernel_ret[i]));
      continue;
    }

    // Slow path when result Tensor is not available synchronously.
    unavailable_kernel_ret = true;
    ret->AndThen([host, handle = handle, ref = kernel_ret[i]]() mutable {
      Tensor& tensor = ref->get<Tensor>();
      handle->emplace<TensorHandle>(host->GetHostDeviceRef(), tensor.metadata(),
                                    AsyncValueRef<Tensor>(std::move(ref)));
    });
  }

  // Keep operands alive if we have unavailable results.
  if (unavailable_kernel_ret)
    RunWhenReady(kernel_ret, [o = RCArray<AsyncValue>(operands.values())] {});
}

static void Execute(Argument<JitExecutable> jit_executable,
                    RepeatedArguments<TensorHandle> operands,
                    RemainingResults results,
                    const ExecutionContext& exec_ctx) {
  // Extract tensors from tensor handle operands to pass them as the compiled
  // kernel arguments.
  llvm::SmallVector<MemrefDesc, 4> memrefs;
  if (auto err = ConvertTensorHandleOperandsToMemrefDesc(operands, &memrefs))
    return ReturnErrors(results, std::move(err));

  // Get an executable that might be specialized to the operands.
  Expected<AsyncValuePtr<Executable>> executable =
      jit_executable->GetExecutable(memrefs);
  if (auto err = executable.takeError())
    return ReturnErrors(results, std::move(err));

  // If executable is available execute it inline.
  if (executable->IsAvailable()) {
    if (executable->IsError()) {
      ReturnErrors(results, executable->GetError());
    } else {
      ExecuteImpl(executable->get(), memrefs, operands, results, exec_ctx);
    }
    return;
  }

  // Otherwise execute it when the executable will become available. This
  // requires careful lifetime extension of all async values passed as operands
  // to the kernel (and also results that will become available asynchronously).

  // Allocate indirect async values for all results, we'll forward them to the
  // actual async values computed by the executable later.
  for (unsigned i = 0; i < results.size(); ++i)
    results.AllocateIndirectResultAt(i);

  // Call executable when it's ready with the original operands.
  executable->AndThen([exec_ctx, executable = *executable,
                       memrefs = std::move(memrefs),
                       r = RCArray<AsyncValue>(results.values()),
                       o = RCArray<AsyncValue>(operands.values())] {
    // Allocate storage for the executable results.
    llvm::SmallVector<RCReference<AsyncValue>> results_storage;
    results_storage.resize(r.size());

    // Reconstruct arguments and results from captured async values.
    RepeatedArguments<TensorHandle> operands(o.values());
    RemainingResults results(results_storage);

    if (executable.IsError()) {
      ReturnErrors(results, executable.GetError());
    } else {
      ExecuteImpl(*executable, memrefs, operands, results, exec_ctx);
    }

    // Forward previously allocated indirect results to the actual results.
    for (unsigned i = 0; i < r.size(); ++i)
      llvm::cast<IndirectAsyncValue>(*r[i]).ForwardTo(
          std::move(results_storage[i]));
  });
}

void RegisterCpuRuntimeCoreRtKernels(KernelRegistry* registry) {
  registry->AddKernel("jitrt.corert.compile", TFRT_KERNEL(Compile));
  registry->AddKernel("jitrt.corert.execute", TFRT_KERNEL(Execute));
}

}  // namespace jitrt
}  // namespace tfrt
