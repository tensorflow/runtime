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
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/mutex.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace {

class OpHandlerRegistry {
 public:
  OpHandler* GetOrNull(string_view name) const {
    return named_op_handlers_.lookup(name);
  }

  void AddOpHandler(std::unique_ptr<OpHandler> op_handler) {
    assert(op_handler);
    all_op_handlers_.emplace_back(std::move(op_handler));
  }

  bool AddNamedOpHandler(string_view name, OpHandler* op_handler) {
    assert(op_handler);
    auto r = named_op_handlers_.try_emplace(name, op_handler);
    (void)r;
    return r.second;
  }

 private:
  // named_op_handlers_ can be looked up via GetOrNull() function.
  llvm::StringMap<OpHandler*> named_op_handlers_;
  std::vector<std::unique_ptr<OpHandler>> all_op_handlers_;
};

class DefaultLocationHandler final : public LocationHandler {
 public:
  explicit DefaultLocationHandler(HostContext* host) : LocationHandler{host} {}

  DecodedLocation DecodeLocation(Location loc) const override {
    return DecodedLocation{};
  }
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
                 std::move(work_queue)),
        default_location_handler_{&context_} {}

  HostContext* GetHostContext() { return &context_; }

  OpHandler* GetOpHandler(string_view name) const {
    return op_handler_registry_.GetOrNull(name);
  }

  void Execute(string_view op_name, OpHandler* op_handler, Location loc,
               MutableArrayRef<TensorHandle> arguments, const OpAttrsRef& attrs,
               MutableArrayRef<TensorHandle> results,
               AsyncValueRef<Chain>* chain);

 private:
  friend class CoreRuntime;

  void SetOpHandlerRegistry(OpHandlerRegistry op_handler_registry) {
    op_handler_registry_ = std::move(op_handler_registry);
  }

  // There is a 1-1 correspondence between HostContext and CoreRuntime.
  HostContext context_;

  OpHandlerRegistry op_handler_registry_;
  const DefaultLocationHandler default_location_handler_;
};

void CoreRuntime::Impl::Execute(string_view op_name, OpHandler* op_handler,
                                Location loc,
                                MutableArrayRef<TensorHandle> arguments,
                                const OpAttrsRef& attrs,
                                MutableArrayRef<TensorHandle> results,
                                AsyncValueRef<Chain>* chain) {
  if (!loc) {
    loc = Location(&default_location_handler_, 0);
  }

  TFRT_TRACE_KERNEL_SCOPE(
      StrCat(op_name, "#op_handler=", op_handler->GetName()));

  // Ask the op_handler to execute the op.  If successful, we're done.
  auto op_handle = op_handler->MakeOp(op_name);
  if (op_handle) {
    op_handle.get()(loc, arguments, attrs, results, chain);
    return;
  }

  // Otherwise, we fail with an 'unknown op' error.
  auto err = loc.EmitErrorAsync("op '" + op_name.str() + "' is not supported");
  for (auto& result : results)
    result = TensorHandle(err.CopyRef(), err.CopyRef());

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
  RegisterStaticKernels(runtime->GetHostContext()->GetRegistry());

  if (op_handler_chains.empty()) return runtime;

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

    if (!op_handler_registry.AddNamedOpHandler(op_handler_chain_name,
                                               fallback)) {
      return MakeStringError("OpHandler ",
                             std::string(op_handler_chain_name).c_str(),
                             " already registered.\n");
    }
  }

  runtime->impl_->SetOpHandlerRegistry(std::move(op_handler_registry));

  return runtime;
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

void CoreRuntime::Execute(string_view op_name, OpHandler* op_handler,
                          Location loc, MutableArrayRef<TensorHandle> arguments,
                          const OpAttrsRef& attrs,
                          MutableArrayRef<TensorHandle> results,
                          AsyncValueRef<Chain>* chain) {
  impl_->Execute(op_name, op_handler, loc, arguments, attrs, results, chain);
}

Expected<CoreRuntimeOp> CoreRuntime::MakeOp(string_view op_name,
                                            OpHandler* op_handler) {
  return op_handler->MakeOp(op_name);
}

}  // namespace tfrt
