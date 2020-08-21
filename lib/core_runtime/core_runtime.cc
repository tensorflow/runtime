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

//===- core_runtime.cc ----------------------------------------------------===//
//
// This file implements the CoreRuntime class.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/core_runtime.h"

#include <string>

#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/op_handler_factory.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/tensor_handle.h"
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

OpHandlerFactory& OpHandlerFactory::GetGlobalOpHandlerFactory() {
  static auto* const global_op_handler_factory = new OpHandlerFactory();
  return *global_op_handler_factory;
}

OpHandler::~OpHandler() {}

class CoreRuntime::Impl {
 public:
  Impl(std::function<void(const DecodedDiagnostic&)> diag_handler,
       std::unique_ptr<HostAllocator> allocator,
       std::unique_ptr<ConcurrentWorkQueue> work_queue)
      : context_(std::move(diag_handler), std::move(allocator),
                 std::move(work_queue)) {}

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
  for (auto& result : results) result = TensorHandle(err.CopyRef());

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
    std::unique_ptr<ConcurrentWorkQueue> work_queue,
    ArrayRef<std::string> op_handler_chains) {
  auto runtime = std::make_unique<CoreRuntime>(
      std::move(diag_handler), std::move(allocator), std::move(work_queue));

  // Register all of the kernels that are statically linked into this executable
  // with our registry.
  RegisterStaticKernels(runtime->GetHostContext()->GetMutableRegistry());

  RegisterTensorConversionFns(runtime->GetHostContext());

  if (op_handler_chains.empty()) return std::move(runtime);

  OpHandlerRegistry op_handler_registry;
  const auto& factory = OpHandlerFactory::GetGlobalOpHandlerFactory();

  OpHandler* null_op_handler;
  if (auto error_or_null_create_fn = factory.Get("null")) {
    const auto& null_create_fn = *error_or_null_create_fn;
    auto error_or_null_op_handler = null_create_fn(runtime.get(), nullptr);
    assert(error_or_null_op_handler);
    null_op_handler = error_or_null_op_handler->get();
    op_handler_registry.AddOpHandler(std::move(*error_or_null_op_handler));
  } else {
    return error_or_null_create_fn.takeError();
  }

  for (string_view op_handler_chain_spec : op_handler_chains) {
    // op_handler_chain_spec is in one of the following two formats:
    // 1) <chain_name>:<op_handler1>|<op_handler2>
    //    Example: cpu:logging|cpu
    // 2) <op_handler1>
    //    If the op_handler chain has only a single op_handler, the chain_name
    //    is optional. Example: cpu

    string_view op_handler_chain_name;
    string_view op_handler_chain;

    // First, split by ':' to get op_handler chain name and the op_handler chain
    // string.
    llvm::SmallVector<string_view, 2> op_handler_name_and_chain;
    op_handler_chain_spec.split(op_handler_name_and_chain, ':');
    if (op_handler_name_and_chain.size() == 1) {
      op_handler_chain = op_handler_name_and_chain[0];
    } else if (op_handler_name_and_chain.size() == 2) {
      op_handler_chain_name = op_handler_name_and_chain[0];
      op_handler_chain = op_handler_name_and_chain[1];
    } else {
      return MakeStringError("Invalid op_handler chain format: ",
                             op_handler_chain_spec);
    }

    // Second, split op_handler_chain by '|' to get op_handler names.
    llvm::SmallVector<string_view, 2> op_handler_names;
    op_handler_chain.split(op_handler_names, '|');

    // OpHandler chain should be created in reverse order. The OpHandler at the
    // end of the chain (e.g., fallbacks) should be created first.
    OpHandler* fallback = null_op_handler;
    for (auto name : llvm::reverse(op_handler_names)) {
      if (auto error_or_create_fn = factory.Get(name)) {
        const auto& create_fn = *error_or_create_fn;
        auto op_handler = create_fn(runtime.get(), fallback);
        if (!op_handler) return op_handler.takeError();
        fallback = op_handler->get();
        op_handler_registry.AddOpHandler(std::move(*op_handler));
      } else {
        return error_or_create_fn.takeError();
      }
    }

    // `fallback` now points to the first op_handler in the op_handler chain.
    if (op_handler_chain_name.empty())
      op_handler_chain_name = fallback->GetName();

    if (!op_handler_registry.AddOpHandlerChain(op_handler_chain_name,
                                               fallback)) {
      return MakeStringError("OpHandler ",
                             std::string(op_handler_chain_name).c_str(),
                             " already registered.\n");
    }
  }

  runtime->impl_->SetOpHandlerRegistry(std::move(op_handler_registry));

  return std::move(runtime);
}

CoreRuntime::CoreRuntime(
    std::function<void(const DecodedDiagnostic&)> diag_handler,
    std::unique_ptr<HostAllocator> allocator,
    std::unique_ptr<ConcurrentWorkQueue> work_queue) {
  // Create the impl for the CoreRuntime, which constructs a HostContext among
  // other things.
  impl_ = std::make_unique<Impl>(std::move(diag_handler), std::move(allocator),
                                 std::move(work_queue));

  auto& ctx =
      GetHostContext()->GetOrCreateSharedContext<CoreRuntimeSharedContext>();
  assert(!ctx.runtime && "cannot already have a CoreRuntime");
  ctx.runtime = this;
}

CoreRuntime::~CoreRuntime() = default;

HostContext* CoreRuntime::GetHostContext() { return impl_->GetHostContext(); }

// Return the CoreRuntime instance that owns the specified HostContext.  This
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
#ifdef TFRT_DISABLE_TRACING
  return op_handler->MakeOp(op_name);
#else  // TFRT_DISABLE_TRACING
  auto op = op_handler->MakeOp(op_name);
  if (!op) return op;
  bool is_fallback = op->IsFallback();
  // TODO(b/155801998): Avoid this string copy.
  return CoreRuntimeOp(
      [op_name = op_name.str(), op = std::move(op.get()),
       op_handler](const OpInvocation& invocation) {
        TFRT_TRACE_KERNEL_SCOPE(
            StrCat(op_name, "#op_handler=", op_handler->GetName()));
        op(invocation);
      },
      is_fallback);
#endif  // TFRT_DISABLE_TRACING
}

Expected<CoreRuntimeOp> CoreRuntime::MakeCompositeOp(const Function* fn) {
  for (auto iter : llvm::enumerate(fn->argument_types().drop_front())) {
    size_t i = iter.index();
    auto& type = iter.value();
    if (type.GetName() != kTensorHandleType) {
      return MakeStringError("The function should only takes type [",
                             kTensorHandleType, "] as input. But the ", i,
                             "-th argument is type [", type.GetName(), "].");
    }
  }
  for (auto iter : llvm::enumerate(fn->result_types().drop_front())) {
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

    SmallVector<AsyncValue*, 4> arguments;
    SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
    arguments.reserve(fn->argument_types().size());
    arguments_ref.reserve(fn->argument_types().size());

    // The first argument is a chain for side-effects.
    if (invocation.chain && *invocation.chain) {
      arguments.push_back(invocation.chain->GetAsyncValue());
    } else {
      arguments_ref.push_back(GetReadyChain(host));
      arguments.push_back(arguments_ref.back().get());
    }

    for (size_t i = 0, e = invocation.arguments.size(); i != e; ++i) {
      arguments_ref.push_back(MakeAvailableAsyncValueRef<TensorHandle>(
          host, invocation.arguments[i].CopyRef()));
      arguments.push_back(arguments_ref.back().get());

      // Clean up the argument to enable input forwarding.
      invocation.arguments[i] = TensorHandle();
    }

    SmallVector<RCReference<AsyncValue>, 4> results;
    results.resize(fn->result_types().size());

    fn->Execute(invocation.exec_ctx, arguments, results);

    // Check if chain is available. If not, wait until the composite op
    // results are fully resolved.
    // TODO(b/161751424) Assess using SyncFunction to execute composite ops.
    if (!results[0]->IsAvailable()) host->Await(results);

    // The first result is the a chain for side-effects.
    if (invocation.chain)
      *invocation.chain = AsyncValueRef<Chain>(std::move(results[0]));

    for (auto iter : llvm::enumerate(llvm::drop_begin(results, 1))) {
      size_t i = iter.index();
      auto& result_av = iter.value();
      if (result_av->IsAvailable()) {
        if (result_av->IsError()) {
          invocation.results[i] =
              TensorHandle(AsyncValueRef<TensorHandle>(result_av.CopyRef()));
        } else {
          assert(result_av->IsType<TensorHandle>());
          invocation.results[i] = result_av->get<TensorHandle>().CopyRef();
        }
      } else {
        llvm_unreachable(
            "composite op must return tensor handler synchronously");
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

    SmallVector<AsyncValue*, 4> arguments;
    SmallVector<RCReference<AsyncValue>, 4> arguments_ref;
    arguments.reserve(fn->argument_types().size());
    arguments_ref.reserve(fn->argument_types().size());

    // The first argument is a chain for side-effects.
    if (invocation.chain && *invocation.chain) {
      arguments.push_back(invocation.chain->GetAsyncValue());
    } else {
      arguments_ref.push_back(GetReadyChain(host));
      arguments.push_back(arguments_ref.back().get());
    }

    for (size_t i = 0, e = invocation.arguments.size(); i != e; ++i) {
      arguments_ref.push_back(invocation.arguments[i].CopyRef());
      arguments.push_back(arguments_ref.back().get());
    }

    SmallVector<RCReference<AsyncValue>, 4> results;
    results.resize(fn->result_types().size());

    fn->Execute(invocation.exec_ctx, arguments, results);

    // Check if chain is available. If not, wait until the native composite op
    // results are fully resolved.
    // TODO(b/161751424) Assess using SyncFunction to execute composite ops.
    if (!results[0]->IsAvailable()) host->Await(results);

    // The first result is the a chain for side-effects.
    if (invocation.chain)
      *invocation.chain = AsyncValueRef<Chain>(std::move(results[0]));

    for (auto iter : llvm::enumerate(llvm::drop_begin(results, 1))) {
      size_t i = iter.index();
      auto& result_av = iter.value();

      invocation.results[i] = result_av.CopyRef();
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
