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

// This file implements the CoreRuntime class.

#include "tfrt/core_runtime/core_runtime.h"

#include <string>

#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {

const char* CoreRuntime::kTensorHandleType = "!corert.tensorhandle";

namespace {

class OpHandlerRegistry {
 public:
  OpHandler* GetOrNull(string_view name) const {
    return op_handers_by_name_.lookup(name);
  }

  void AddOpHandler(std::unique_ptr<OpHandler> op_handler) {
    assert(op_handler);
    all_op_handlers_.emplace_back(std::move(op_handler));
  }

  bool AddOpHandlerChain(string_view name, OpHandler* op_handler) {
    assert(op_handler);
    auto r = op_handers_by_name_.try_emplace(name, op_handler);
    (void)r;
    return r.second;
  }

 private:
  // op_handers_by_name_ can be looked up via GetOrNull() function.
  llvm::StringMap<OpHandler*> op_handers_by_name_;
  std::vector<std::unique_ptr<OpHandler>> all_op_handlers_;
};

}  // namespace

OpHandler::~OpHandler() {}

class CoreRuntime::Impl {
 public:
  Impl(std::function<void(const DecodedDiagnostic&)> diag_handler,
       std::unique_ptr<HostAllocator> allocator,
       std::unique_ptr<ConcurrentWorkQueue> work_queue,
       string_view host_device_name)
      : context_(std::move(diag_handler), std::move(allocator),
                 std::move(work_queue), host_device_name) {}

  HostContext* GetHostContext() { return &context_; }

  OpHandler* GetOpHandler(string_view name) const {
    return op_handler_registry_.GetOrNull(name);
  }

  void Execute(const ExecutionContext& exec_ctx, string_view op_name,
               OpHandler* op_handler, MutableArrayRef<TensorHandle> arguments,
               const OpAttrsRef& attrs, MutableArrayRef<TensorHandle> results,
               AsyncValueRef<Chain>* chain);

  void TakeOpHandler(std::unique_ptr<OpHandler> op_handler) {
    op_handler_registry_.AddOpHandler(std::move(op_handler));
  }

  void RegisterOpHandlerChain(string_view name, OpHandler* op_handler) {
    op_handler_registry_.AddOpHandlerChain(name, op_handler);
  }

 private:
  friend class CoreRuntime;

  void SetOpHandlerRegistry(OpHandlerRegistry op_handler_registry) {
    op_handler_registry_ = std::move(op_handler_registry);
  }

  // There is a 1-1 correspondence between HostContext and CoreRuntime.
  HostContext context_;

  OpHandlerRegistry op_handler_registry_;
};

void CoreRuntime::Impl::Execute(const ExecutionContext& exec_ctx,
                                string_view op_name, OpHandler* op_handler,
                                MutableArrayRef<TensorHandle> arguments,
                                const OpAttrsRef& attrs,
                                MutableArrayRef<TensorHandle> results,
                                AsyncValueRef<Chain>* chain) {
  // Ask the op_handler to execute the op.  If successful, we're done.
  auto op_handle = op_handler->MakeOp(op_name);
  if (op_handle) {
    op_handle.get()(exec_ctx, arguments, attrs, results, chain);
    return;
  }

  // Otherwise, we fail with an 'unknown op' error.
  auto err =
      EmitErrorAsync(exec_ctx, "op '" + op_name.str() + "' is not supported");
  for (auto& result : results) result = TensorHandle(err);

  if (chain) *chain = std::move(err);
}

//===----------------------------------------------------------------------===//
// Constructor / Destructor Logic
//===----------------------------------------------------------------------===//

namespace {
// This struct allows HostContext to keep an upwards pointer to the containing
// CoreRuntime.  This is all maintained internally to CoreRuntime, external
// clients should just use the CoreRuntime::GetFromHostContext static method.
struct CoreRuntimeSharedContext : public SharedContext {
  explicit CoreRuntimeSharedContext(HostContext* host) {}
  CoreRuntime* runtime = nullptr;
};
}  // namespace

llvm::Expected<std::unique_ptr<CoreRuntime>> CoreRuntime::Create(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue) {
  return CoreRuntime::Create(std::move(diag_handler), std::move(allocator),
                             std::move(work_queue),
                             HostContext::kDefaultHostDeviceName);
}
llvm::Expected<std::unique_ptr<CoreRuntime>> CoreRuntime::Create(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue,
    string_view host_device_name) {
  auto runtime = std::make_unique<CoreRuntime>(
      std::move(diag_handler), std::move(allocator), std::move(work_queue),
      host_device_name);

  // Register all of the kernels that are statically linked into this
  // executable with our registry.
  RegisterStaticKernels(runtime->GetHostContext()->GetMutableRegistry());

  RegisterTensorConversionFns(runtime->GetHostContext());
  return std::move(runtime);
}

CoreRuntime::CoreRuntime(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue,
    string_view host_device_name) {
  // Create the impl for the CoreRuntime, which constructs a HostContext
  // among other things.
  impl_ = std::make_unique<Impl>(std::move(diag_handler), std::move(allocator),
                                 std::move(work_queue), host_device_name);

  auto& ctx =
      GetHostContext()->GetOrCreateSharedContext<CoreRuntimeSharedContext>();
  assert(!ctx.runtime && "cannot already have a CoreRuntime");
  ctx.runtime = this;
}

CoreRuntime::~CoreRuntime() = default;

HostContext* CoreRuntime::GetHostContext() { return impl_->GetHostContext(); }

// Return the CoreRuntime instance that owns the specified HostContext. This
// returns null if the specified HostContext isn't owned by a CoreRuntime.
CoreRuntime* CoreRuntime::GetFromHostContext(HostContext* context) {
  return context->GetOrCreateSharedContext<CoreRuntimeSharedContext>().runtime;
}

//===----------------------------------------------------------------------===//
// Other
//===----------------------------------------------------------------------===//

OpHandler* CoreRuntime::GetOpHandler(string_view name) const {
  return impl_->GetOpHandler(name);
}

void CoreRuntime::Execute(const ExecutionContext& exec_ctx, string_view op_name,
                          OpHandler* op_handler,
                          MutableArrayRef<TensorHandle> arguments,
                          const OpAttrsRef& attrs,
                          MutableArrayRef<TensorHandle> results,
                          AsyncValueRef<Chain>* chain) {
  impl_->Execute(exec_ctx, op_name, op_handler, arguments, attrs, results,
                 chain);
}

Expected<CoreRuntimeOp> CoreRuntime::MakeOp(string_view op_name,
                                            OpHandler* op_handler) {
  auto op = op_handler->MakeOp(op_name);
  if (!tracing::IsTracingEnabled(tracing::TracingLevel::Default)) return op;
  if (!op) return op;
  bool is_fallback = op->IsFallback();
  auto device = op->GetDeviceRef();
  // TODO(b/155801998): Avoid this string copy.
  return CoreRuntimeOp(
      [op_name = op_name.str(), op = std::move(op.get()),
       op_handler](const OpInvocation& invocation) {
        TFRT_TRACE_SCOPE(
            Default,
            StrCat(op_name, "#op_handler=", op_handler->GetName(), "#"));
        op(invocation);
      },
      is_fallback, std::move(device), op->GetTensorType());
}

Expected<CoreRuntimeOp> CoreRuntime::MakeCompositeOp(const Function* fn) {
  for (const auto& iter : llvm::enumerate(fn->argument_types().drop_front())) {
    size_t i = iter.index();
    auto& type = iter.value();
    if (type.GetName() != kTensorHandleType) {
      return MakeStringError("The function should only takes type [",
                             kTensorHandleType, "] as input. But the ", i,
                             "-th argument is type [", type.GetName(), "].");
    }
  }
  for (const auto& iter : llvm::enumerate(fn->result_types().drop_front())) {
    size_t i = iter.index();
    auto& type = iter.value();
    if (type.GetName() != kTensorHandleType) {
      return MakeStringError("The function should only returns type [",
                             kTensorHandleType, "]. But the ", i,
                             "-th results is type [", type.GetName(), "].");
    }
  }
  auto execute_fn = [fn = fn](const OpInvocation& invocation) {
    auto* host = invocation.exec_ctx.host();

    // TODO(fishx): Return an error to the client instead of asserting.
    assert(invocation.arguments.size() + 1 == fn->argument_types().size());
    assert(invocation.results.size() + 1 == fn->result_types().size());

    llvm::SmallVector<AsyncValue*, 4> arguments;
    llvm::SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
    arguments.reserve(fn->argument_types().size());
    arguments_ref.reserve(fn->argument_types().size());

    // The first argument is a chain for side-effects.
    if (invocation.chain && *invocation.chain) {
      arguments.push_back(invocation.chain->GetAsyncValue());
    } else {
      arguments_ref.push_back(GetReadyChain());
      arguments.push_back(arguments_ref.back().get());
    }

    for (size_t i = 0, e = invocation.arguments.size(); i != e; ++i) {
      arguments_ref.push_back(MakeAvailableAsyncValueRef<TensorHandle>(
          host, invocation.arguments[i].CopyRef()));
      arguments.push_back(arguments_ref.back().get());

      // Clean up the argument to enable input forwarding.
      invocation.arguments[i] = TensorHandle();
    }

    llvm::SmallVector<RCReference<AsyncValue>, 4> results;
    results.resize(fn->result_types().size());

    fn->Execute(invocation.exec_ctx, arguments, results);

    // The first result is the a chain for side-effects.
    if (invocation.chain)
      *invocation.chain = AsyncValueRef<Chain>(std::move(results[0]));

    for (const auto& iter : llvm::enumerate(llvm::drop_begin(results, 1))) {
      size_t i = iter.index();
      auto& result_av = iter.value();
      if (result_av->IsAvailable()) {
        if (result_av->IsError()) {
          invocation.results[i] =
              TensorHandle(AsyncValueRef<TensorHandle>(std::move(result_av)));
        } else {
          assert(result_av->IsType<TensorHandle>());
          invocation.results[i] = result_av->get<TensorHandle>().CopyRef();
        }
      } else {
        auto device_av =
            MakeUnconstructedAsyncValueRef<RCReference<Device>>(host);
        auto metadata_av = MakeUnconstructedAsyncValueRef<TensorMetadata>(host);
        auto tensor_ind_av = MakeIndirectAsyncValue(host);

        result_av->AndThen([result_av = result_av,
                            device_av = device_av.CopyRef(),
                            metadata_av = metadata_av.CopyRef(),
                            tensor_ind_av = tensor_ind_av]() mutable {
          if (result_av->IsError()) {
            device_av.SetError(result_av->GetError());
            metadata_av.SetError(result_av->GetError());
            tensor_ind_av->SetError(result_av->GetError());
            return;
          }
          auto& th = result_av->get<TensorHandle>();

          if (th.IsDeviceAvailable()) {
            device_av.emplace(th.GetAvailableDevice());
          } else {
            th.GetAsyncDevice().AndThen(
                [th_device = th.GetAsyncDevice().CopyRef(),
                 device_av = std::move(device_av)]() {
                  if (th_device.IsError()) {
                    device_av.SetError(th_device.GetError());
                  } else {
                    device_av.emplace(th_device.get());
                  }
                });
          }

          if (th.IsMetadataAvailable()) {
            metadata_av.emplace(th.GetAvailableMetadata());
          } else {
            th.GetAsyncMetadata().AndThen(
                [th_metadata = th.GetAsyncMetadata().CopyRef(),
                 metadata_av = std::move(metadata_av)]() {
                  if (th_metadata.IsError()) {
                    metadata_av.SetError(th_metadata.GetError());
                  } else {
                    metadata_av.emplace(th_metadata.get());
                  }
                });
          }

          tensor_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
        });

        invocation.results[i] =
            TensorHandle(std::move(device_av), std::move(metadata_av),
                         AsyncValueRef<Tensor>(std::move(tensor_ind_av)));
      }
    }
  };
  return CoreRuntimeOp(std::move(execute_fn), false);
}

Expected<CoreRuntimeOp> CoreRuntime::MakeNativeCompositeOp(const Function* fn) {
  auto execute_fn = [fn = fn](const CompositeOpInvocation& invocation) {
    auto* host = invocation.exec_ctx.host();

    // TODO(fishx): Return an error to the client instead of asserting.
    if (invocation.arguments.size() + 1 != fn->argument_types().size()) {
      TFRT_LOG(FATAL) << "Fn has " << fn->argument_types().size()
                      << " arguments, while invocation provides "
                      << invocation.arguments.size() << " arguments.";
    }
    if (invocation.results.size() + 1 != fn->result_types().size()) {
      TFRT_LOG(FATAL) << "Fn has " << fn->result_types().size()
                      << " results, while invocation provides "
                      << invocation.results.size() << " results.";
    }

    llvm::SmallVector<AsyncValue*, 4> arguments;
    llvm::SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
    arguments.reserve(fn->argument_types().size());
    arguments_ref.reserve(fn->argument_types().size());

    // The first argument is a chain for side-effects.
    if (invocation.chain && *invocation.chain) {
      arguments.push_back(invocation.chain->GetAsyncValue());
    } else {
      arguments_ref.push_back(GetReadyChain());
      arguments.push_back(arguments_ref.back().get());
    }

    for (size_t i = 0, e = invocation.arguments.size(); i != e; ++i) {
      arguments_ref.push_back(invocation.arguments[i]);
      arguments.push_back(arguments_ref.back().get());
    }

    llvm::SmallVector<RCReference<AsyncValue>, 4> results;
    results.resize(fn->result_types().size());

    fn->Execute(invocation.exec_ctx, arguments, results);

    // Check if chain is available. If not, wait until the native composite
    // op results are fully resolved.
    // TODO(b/161751424) Assess using SyncFunction to execute composite ops.
    if (!results[0]->IsAvailable()) host->Await(results);

    // The first result is the a chain for side-effects.
    if (invocation.chain)
      *invocation.chain = AsyncValueRef<Chain>(std::move(results[0]));

    for (const auto& iter : llvm::enumerate(llvm::drop_begin(results, 1))) {
      size_t i = iter.index();
      auto& result_av = iter.value();

      invocation.results[i] = result_av;
    }
  };
  return CoreRuntimeOp(std::move(execute_fn));
}

void CoreRuntime::TakeOpHandler(std::unique_ptr<OpHandler> op_handler) {
  impl_->TakeOpHandler(std::move(op_handler));
}

void CoreRuntime::RegisterOpHandler(string_view name, OpHandler* op_handler) {
  impl_->RegisterOpHandlerChain(name, op_handler);
}

}  // namespace tfrt
