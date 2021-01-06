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

//===- cpurt_kernels.cc - CPURT kernels -----------------------------------===//
//
// This file defines the C++ kernels for the CPURT dialect.
//
//===----------------------------------------------------------------------===//

#include <sys/types.h>

#include <cstdint>
#include <memory>

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/cpu/jit/cpurt.h"
#include "tfrt/dtype/dtype.h"
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
  if (kernel.nested_symbols().size() != 1) {
    return EmitErrorAsync(
        exec_ctx, "compiled kernel must be referenced by one nested symbol");
  }

  ResourceContext* res_ctx = exec_ctx.resource_context();
  auto* compilation_cache =
      res_ctx->GetOrCreateResource<CompilationResultCache>("cpurt.cache", host);

  // TODO(ezhulenev): Compute cache key based on the content of MLIR module.
  intptr_t key = exec_ctx.location().data;

  // Return compiled kernel from the cache.
  if (auto compiled = compilation_cache->Find(key)) return compiled;

  // Create an async task for each worker thread.
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
// Execute compiled CPURT kernels.
// -------------------------------------------------------------------------- //

static AsyncValueRef<Chain> Execute(
    Argument<CompilationResult> compilation_result, Argument<Chain> in_chain,
    RepeatedArguments<Tensor> operands, const ExecutionContext& exec_ctx) {
  auto chain = compilation_result->Execute(operands, exec_ctx);
  // Keep operands alive until the execution is completed.
  chain.AndThen([operands = RCArray<AsyncValue>(operands.values())] {});
  return chain;
}

// -------------------------------------------------------------------------- //
// Execute compiled CPURT kernels with CoreRT interop.
// -------------------------------------------------------------------------- //

// TODO(ezhulenev): Split this file into cpurt library and multiple kernel
// libraries for different execution contexts: graph, op-by-op, fallback, etc...
static void CoreRtExecute(Argument<CompilationResult> compilation_result,
                          RemainingArguments args, RemainingResults results,
                          DenseAttr operand_sizes,
                          const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  // Extract tensor handle and tensor shape operands.
  DenseView operand_sizes_view = CreateDenseView(operand_sizes);
  ArrayRef<int32_t> operand_sizes_flat = operand_sizes_view.GetFlat<int32_t>();

  assert(operand_sizes_flat.size() == 3);
  int32_t num_tensor_handle_args = operand_sizes_flat[1];
  int32_t num_shape_args = operand_sizes_flat[2];

  // Split arguments into handles and shapes.
  RepeatedArguments<TensorHandle> tensor_handles(
      args.values().take_front(num_tensor_handle_args));
  RepeatedArguments<TensorShape> tensor_shapes(
      args.values().drop_front(num_tensor_handle_args));
  assert(results.size() == tensor_shapes.size());

  // Allocate TensorHandles for all results based on the shape arguments.
  for (int i = 0; i < tensor_shapes.size(); ++i) {
    TensorShape& shape = tensor_shapes[i];

    // TODO(ezhulenev): Infer result dtype from the compiled kernel signature,
    // for now always assume F32 data type.
    TensorMetadata metadata(DType(DType::F32), shape);

    auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(metadata, host);
    results.AllocateAt<TensorHandle>(i)->emplace<TensorHandle>(
        host->GetHostDeviceRef(), metadata, std::move(dht));
  }

  // Convert all tensor handle arguments and results to tensors (buffers) that
  // will be passed to the compiled kernel.
  SmallVector<AsyncValue*, 4> tensor_operands;

  // Process all TensorHandle arguments.
  for (int i = 0; i < num_tensor_handle_args; ++i) {
    TensorHandle& hdl = args[i]->get<TensorHandle>();
    tensor_operands.push_back(hdl.GetAsyncTensor());
  }

  // Process all TensorHandle results.
  for (int i = 0; i < num_shape_args; ++i) {
    TensorHandle& hdl = results[i]->get<TensorHandle>();
    tensor_operands.push_back(hdl.GetAsyncTensor());
  }

  // Call compiled kernel with tensor operands.
  auto chain = compilation_result->Execute(
      RepeatedArguments<Tensor>(tensor_operands), exec_ctx);

  // Keep args and results alive until the execution is completed.
  chain.AndThen([args = RCArray<AsyncValue>(args.values()),
                 results = RCArray<AsyncValue>(results.values())]() {
    for (int i = 0; i < results.size(); ++i) {
      AsyncValue* result = results[i]->get<TensorHandle>().GetAsyncTensor();
      result->SetStateConcrete();
    }
  });
}

void RegisterCpuRuntimeKernels(KernelRegistry* registry) {
  registry->AddKernel("cpurt.compile", TFRT_KERNEL(Compile));
  registry->AddKernel("cpurt.execute", TFRT_KERNEL(Execute));
  registry->AddKernel("cpurt.corert.execute", TFRT_KERNEL(CoreRtExecute));
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
