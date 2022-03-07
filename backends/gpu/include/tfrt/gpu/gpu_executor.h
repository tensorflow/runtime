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

// Declaration of functions to execute a tfrt_gpu program.

#ifndef TFRT_GPU_GPU_EXECUTOR_H_
#define TFRT_GPU_GPU_EXECUTOR_H_

#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/dense_map_utils.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"

// This file provides the gpu executor API by wrapping some BEF executor
// functionality and checking/using the assumptions of a gpu executable.
//
// That is, the entry function name is hard-coded in the BEF file, and the entry
// function has the following signature:
//
//   !tfrt.chain, !tfrt_gpu.stream, !tfrt_gpu.buffer* -> !tfrt.chain
//
namespace tfrt {
class BEFFile;
namespace gpu {

// Creates and caches GpuContexts and an associated tfrt::ResourceContexts.
class GpuContextCache {
  using Resources = std::pair<AsyncValueRef<GpuContext>, ResourceContext*>;

 public:
  GpuContextCache() = default;
  GpuContextCache(GpuContextCache&&) = default;
  GpuContextCache& operator=(GpuContextCache&&) = default;
  ~GpuContextCache();

  // Returns the `context` as non-owning AsyncValue plus a resource context
  // that is unique to `context`.
  Resources GetOrCreate(wrapper::Context context);

 private:
  llvm::SmallDenseMap<wrapper::Context, Resources> context_resources_;
};

struct BorrowedStreamDeleter {
  using pointer = AsyncValuePtr<GpuStream>;
  void operator()(pointer ptr);
};
using BorrowedStream = std::unique_ptr<void, BorrowedStreamDeleter>;
// Returns the `stream` belonging to `context` as a non-owning AsyncValue.
BorrowedStream MakeBorrowedStream(AsyncValueRef<GpuContext> context,
                                  wrapper::Stream stream);

// Opens a BEF file from `path` as appropriately aligned MemoryBuffer.
// Use '-' to read from stdin.
using UniqueMemoryBuffer = std::unique_ptr<llvm::MemoryBuffer>;
llvm::Expected<UniqueMemoryBuffer> OpenBefBuffer(llvm::StringRef path);

// Creates a diagnostic handler to pass to host context.
using DiagHandler = std::function<void(const DecodedDiagnostic&)>;
DiagHandler GetDiagHandler(mlir::MLIRContext* context);
DiagHandler GetDiagHandler();  // Creates a MLIRContext internally.

// Creates an execution context suitable for gpu execution.
llvm::Expected<ExecutionContext> CreateExecutionContext(
    HostContext* host, ResourceContext* resource_ctx);

// Struct of gpu executor metadata stored in BEF. The data must have been added
// to the mlir module by a call to `setEntryPoint()` before lowering to BEF.
// The data can be retrieved by a call to `GetEntryPoint()`.
struct EntryPoint {
  wrapper::Platform platform;
  std::string function_name;
  std::vector<int64_t> buffer_sizes;
};
llvm::Expected<EntryPoint> GetEntryPoint(const BEFFile& file,
                                         const ExecutionContext& exec_ctx);

// Acquires the primary gpu context and creates a stream for the given gpu.
AsyncValueRef<GpuStream> CreateGpuStream(wrapper::Platform platform,
                                         int ordinal = 0);

// Creates a buffer containing gpu device memory of the given size.
AsyncValueRef<GpuBuffer> AllocateGpuBuffer(const GpuStream& stream,
                                           size_t size_bytes);

// Executes a BEF function using the provided arguments.
AsyncValueRef<Chain> Execute(const ExecutionContext& exec_ctx,
                             const Function& function,
                             AsyncValueRef<Chain> chain,
                             AsyncValueRef<GpuStream> stream,
                             ArrayRef<AsyncValueRef<GpuBuffer>> buffers);

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_GPU_EXECUTOR_H_
