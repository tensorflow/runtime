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

// GPU Executor Test Driver

#include <cstdlib>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/gpu/gpu_executor.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/support/error_util.h"

// Loads a BEF file from `path`, retrieves the entry point, creates a stream
// on device 0's primary context, allocates (but does not initialize) buffer
// arguments, executes the entry point function and synchronizes the stream.
static llvm::Error ExecuteBefFile(llvm::StringRef path) {
  mlir::MLIRContext context;
  tfrt::HostContext host_ctx(
      tfrt::gpu::GetDiagHandler(&context), tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/32, /*num_blocking_threads=*/16));
  tfrt::RegisterStaticKernels(host_ctx.GetMutableRegistry());

  auto buffer = tfrt::gpu::OpenBefBuffer(path);
  if (!buffer) return buffer.takeError();
  mlir::ArrayRef<uint8_t> data(
      reinterpret_cast<const uint8_t*>(buffer.get()->getBufferStart()),
      buffer.get()->getBufferSize());
  auto file =
      tfrt::BEFFile::Open(data, host_ctx.GetKernelRegistry(),
                          host_ctx.diag_handler(), host_ctx.allocator());
  if (!file) return tfrt::MakeStringError("Failed to open file");

  // Augment diagnostics with source and verify expected errors.
  llvm::SourceMgr src_mgr;
  mlir::SourceMgrDiagnosticVerifierHandler handler(src_mgr, &context);

  tfrt::ResourceContext resource_ctx;
  auto exec_ctx = tfrt::gpu::CreateExecutionContext(&host_ctx, &resource_ctx);
  if (!exec_ctx) return exec_ctx.takeError();
  auto entry_point = tfrt::gpu::GetEntryPoint(*file, *exec_ctx);
  if (!entry_point) return entry_point.takeError();

  const tfrt::Function* function =
      file->GetFunction(entry_point->function_name);
  if (!function)
    return tfrt::MakeStringError(entry_point->function_name,
                                 " function not found");

  tfrt::AsyncValueRef<tfrt::Chain> chain = tfrt::GetReadyChain();
  tfrt::AsyncValueRef<tfrt::gpu::GpuStream> stream =
      tfrt::gpu::CreateGpuStream(entry_point->platform);
  if (stream.IsError()) return tfrt::MakeStringError(stream.GetError());
  mlir::SmallVector<tfrt::AsyncValueRef<tfrt::gpu::GpuBuffer>, 4> buffers;
  llvm::transform(entry_point->buffer_sizes, std::back_inserter(buffers),
                  [&](int64_t size_bytes) {
                    return AllocateGpuBuffer(*stream, size_bytes);
                  });

  tfrt::AsyncValueRef<tfrt::Chain> result =
      Execute(*exec_ctx, *function, chain, stream, buffers);
  tfrt::Await(result.GetAsyncValue());

  // The above only guarantees that work depending on `result` has completed.
  // Flush all remaining work before checking for potential errors.
  host_ctx.Quiesce();

  if (failed(handler.verify()))
    return tfrt::MakeStringError("unexpected errors reported");

  if (result.IsError())
    return tfrt::MakeStringError("result has error: ", result.GetError());

  // Synchronize the stream.
  return tfrt::gpu::wrapper::StreamSynchronize(stream->get());
}

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "GPU Executor test driver\n");

  if (auto error = ExecuteBefFile(input_filename)) {
    llvm::errs() << toString(std::move(error)) << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
