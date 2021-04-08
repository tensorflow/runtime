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

// This file implements the CoreRuntimeCpuDriver.
#include "driver.h"

#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"
#include "tfrt/cpu/core_runtime/null_op_handler.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace example {
constexpr const char* kCpuOpHandlerName = "cpu";

static std::unique_ptr<CoreRuntime> CreateCoreRuntime() {
  auto diag_handler = [](const DecodedDiagnostic& diag) {
    llvm::errs() << "Encountered runtime error: " << diag.message << "\n";
  };
  auto corert =
      CoreRuntime::Create(diag_handler, tfrt::CreateMallocAllocator(),
                          tfrt::CreateMultiThreadedWorkQueue(
                              /*num_threads=*/4, /*num_blocking_threads=*/64));
  auto null_op_handler = tfrt::CreateNullOpHandler(corert->get());

  auto cpu_device = corert.get()->GetHostContext()->GetHostDeviceRef();
  auto cpu_op_handler = tfrt::CreateCpuOpHandler(
      corert->get(), std::move(cpu_device), null_op_handler.get());

  corert.get()->RegisterOpHandler(kCpuOpHandlerName, cpu_op_handler.get());

  if (!corert) {
    TFRT_LOG(FATAL) << corert.takeError();
  }
  return std::move(corert.get());
}

CoreRuntimeCpuDriver::CoreRuntimeCpuDriver()
    : CoreRuntimeCpuDriver(CreateCoreRuntime()) {}

CoreRuntimeCpuDriver::CoreRuntimeCpuDriver(std::unique_ptr<CoreRuntime> corert)
    : corert_(std::move(corert)),
      op_handler_(corert_->GetOpHandler(kCpuOpHandlerName)),
      chain_(MakeAvailableAsyncValueRef<Chain>(corert_->GetHostContext())) {
  assert(op_handler_);
}

void CoreRuntimeCpuDriver::Execute(string_view op_name,
                                   MutableArrayRef<TensorHandle> args,
                                   const OpAttrsRef& attrs,
                                   MutableArrayRef<TensorHandle> results) {
  Expected<RCReference<RequestContext>> req_ctx =
      tfrt::RequestContextBuilder(GetHostContext(), &resource_context_).build();
  if (!req_ctx)
    TFRT_LOG(FATAL) << "Failed to build a RequestContext: "
                    << req_ctx.takeError();
  Execute(ExecutionContext(std::move(*req_ctx)), op_name, args, attrs, results);
}

void CoreRuntimeCpuDriver::Execute(const ExecutionContext& exec_ctx,
                                   string_view op_name,
                                   MutableArrayRef<TensorHandle> args,
                                   const OpAttrsRef& attrs,
                                   MutableArrayRef<TensorHandle> results) {
  corert_->Execute(exec_ctx, op_name, op_handler_, args, attrs, results,
                   &chain_);
}

CoreRuntimeOp CoreRuntimeCpuDriver::MakeOp(string_view op_name) {
  auto handle = corert_->MakeOp(op_name, op_handler_);
  assert(handle);
  return std::move(handle.get());
}

CoreRuntimeOp CoreRuntimeCpuDriver::MakeCompositeOp(const Function* fn) {
  Expected<CoreRuntimeOp> handle = corert_->MakeCompositeOp(fn);
  if (!handle) {
    TFRT_LOG(FATAL) << handle.takeError();
  }
  return std::move(handle.get());
}

CoreRuntimeOp CoreRuntimeCpuDriver::MakeNativeCompositeOp(const Function* fn) {
  Expected<CoreRuntimeOp> handle = corert_->MakeNativeCompositeOp(fn);
  if (!handle) {
    TFRT_LOG(FATAL) << handle.takeError();
  }
  return std::move(handle.get());
}

void CoreRuntimeCpuDriver::WaitForHostContextQuiesce() {
  corert_->GetHostContext()->Quiesce();
}

ExecutionContext CoreRuntimeCpuDriver::CreateExecutionContext(
    const char* filename, int line_number) {
  locations_.push_back({filename, line_number});

  auto req_ctx =
      tfrt::RequestContextBuilder(GetHostContext(), &resource_context_).build();
  if (!req_ctx)
    TFRT_LOG(FATAL) << "Failed to build a RequestContext: "
                    << req_ctx.takeError();
  Location location(this, /*data=*/locations_.size() - 1);

  return ExecutionContext{std::move(*req_ctx), location};
}

DecodedLocation CoreRuntimeCpuDriver::DecodeLocation(Location loc) const {
  // TODO(b/147635252): Need a mutex to protect locations_.
  return DecodedLocation{locations_[loc.data].first,
                         locations_[loc.data].second};
}
}  // namespace example
}  // namespace tfrt
