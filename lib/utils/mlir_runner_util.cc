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

#include "tfrt/utils/mlir_runner_util.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "llvm/ADT/None.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "tfrt/bef_converter/mlir_to_bef.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"

namespace tfrt {
namespace testing {
namespace {

std::unique_ptr<HostContext> CreateHostContext() {
  auto decoded_diagnostic_handler = [&](const DecodedDiagnostic& diag) {
    TFRT_LOG(FATAL) << "Encountered error while executing, aborting: "
                    << diag.message;
  };
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();
  std::unique_ptr<HostAllocator> host_allocator = CreateMallocAllocator();
  auto host_ctx = std::make_unique<HostContext>(decoded_diagnostic_handler,
                                                std::move(host_allocator),
                                                std::move(work_queue));
  RegisterStaticKernels(host_ctx->GetMutableRegistry());
  return host_ctx;
}

}  // namespace

TfrtMlirRunner::Builder::Builder() : host_context_(CreateHostContext()) {}

TfrtMlirRunner TfrtMlirRunner::Builder::Compile() {
  assert(!mlir_input_.empty() &&
         "mlir_input must be set before calling Compile.");
  assert(!fn_name_.empty() && "fn_name must be set before calling Compile.");
  assert(mlir_context_ && "MLIR context must be set before calling Compile.");

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString(mlir_input_, mlir_context_);

  tfrt::BefBuffer bef_buffer =
      tfrt::ConvertMLIRToBEF(module.get(), /*disable_optional_sections=*/true);
  auto bef_file =
      BEFFile::Open(bef_buffer, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  assert((bef_file->GetFunction(fn_name_)->num_arguments() == inputs_.size()) &&
         "Incorrect number of arguments set.");
  return TfrtMlirRunner(fn_name_, std::move(bef_buffer), std::move(inputs_));
}

TfrtMlirRunner::TfrtMlirRunner(const std::string& fn_name, BefBuffer bef_buffer,
                               std::vector<RCReference<AsyncValue>> inputs)
    : fn_name_(fn_name),
      inputs_(std::move(inputs)),
      bef_buffer_(bef_buffer),
      host_context_(CreateHostContext()),
      execution_context_(*tfrt::RequestContextBuilder(host_context_.get(),
                                                      resource_context_.get())
                              .build()) {
  input_ptrs_.resize(inputs_.size());
  std::transform(inputs_.begin(), inputs_.end(), input_ptrs_.begin(),
                 [](auto& value) { return value.get(); });
  bef_file_ =
      BEFFile::Open(bef_buffer_, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  func_ = bef_file_->GetFunction(fn_name_);
}

llvm::SmallVector<RCReference<AsyncValue>> TfrtMlirRunner::Run() {
  llvm::SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(func_->result_types().size());
  func_->Execute(execution_context_, input_ptrs_, results);
  return std::move(results);
}

}  // namespace testing
}  // namespace tfrt
