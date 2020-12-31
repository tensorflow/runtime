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

//===- async_runtime.cc -----------------------------------------*- C++ -*-===//
//
// MLIR Async Runtime implemented on top of TFRT HostContext and host
// concurrency primitives.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/jit/async_runtime.h"

#include <memory>

#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/support/concurrent_vector.h"
#include "tfrt/support/ref_count.h"

// -------------------------------------------------------------------------- //
// Define AsyncToken and AsyncGroup in the mlir::runtime namespace to implement
// opaque structs defined in the MLIR Async Runtime API header file.
// -------------------------------------------------------------------------- //

namespace mlir {
namespace runtime {

class AsyncToken : public ::tfrt::cpu::jit::AsyncRuntimeObject {
  using AsyncRuntime = ::tfrt::cpu::jit::AsyncRuntime;

 public:
  explicit AsyncToken(tfrt::AsyncValueRef<tfrt::Chain> chain)
      : chain_(std::move(chain)) {}

  ::tfrt::AsyncValue* GetAsyncValue() const { return chain_.GetAsyncValue(); }

 private:
  tfrt::AsyncValueRef<tfrt::Chain> chain_;
};

class AsyncGroup : public ::tfrt::cpu::jit::AsyncRuntimeObject {
  using AsyncTokens = ::tfrt::ConcurrentVector<AsyncToken*>;
  using AsyncRuntime = ::tfrt::cpu::jit::AsyncRuntime;

 public:
  explicit AsyncGroup(AsyncRuntime* runtime)
      : runtime_(runtime),
        async_tokens_(std::make_unique<AsyncTokens>(/*initial_capacity=*/16)) {}

  ~AsyncGroup() override {
    for (auto* obj : async_tokens_->ToArrayRef()) runtime_->DropRef(obj);
  }

  size_t AddToken(AsyncToken* token) {
    AsyncRuntime::AddRef(token);  // keep token alive while *this is alive
    return async_tokens_->emplace_back(token);
  }

  size_t size() const { return async_tokens_->size(); }

  llvm::SmallVector<::tfrt::AsyncValue*, 4> GetAsyncValues() const {
    auto tokens = llvm::map_range(
        async_tokens_->ToArrayRef(),
        [](AsyncToken* token) { return token->GetAsyncValue(); });
    return {tokens.begin(), tokens.end()};
  }

  llvm::ArrayRef<AsyncToken*> GetAsyncTokens() const {
    return async_tokens_->ToArrayRef();
  }

 private:
  AsyncRuntime* runtime_;
  std::unique_ptr<AsyncTokens> async_tokens_;
};

}  // namespace runtime
}  // namespace mlir

// -------------------------------------------------------------------------- //

namespace tfrt {
namespace cpu {
namespace jit {

using AsyncToken = AsyncRuntime::AsyncToken;
using AsyncGroup = AsyncRuntime::AsyncGroup;

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncToken* token) {
  return token->GetAsyncValue();
}

/*static*/ SmallVector<AsyncValue*, 4> AsyncRuntime::GetAsyncValues(
    AsyncGroup* group) {
  return group->GetAsyncValues();
}

/*static*/ void AsyncRuntime::AddRef(AsyncRuntimeObject* obj, unsigned count) {
  assert(count == 1 && "tfrt::ReferenceCounted can add just one ref");
  obj->AddRef();
}

/*static*/ void AsyncRuntime::DropRef(AsyncRuntimeObject* obj, unsigned count) {
  assert(count == 1 && "tfrt::ReferenceCounted can drop just one ref");
  obj->DropRef();
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncToken* token) {
  return static_cast<AsyncRuntimeObject*>(token);
}
/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncGroup* group) {
  return static_cast<AsyncRuntimeObject*>(group);
}

AsyncToken* AsyncRuntime::CreateToken() {
  auto chain = MakeConstructedAsyncValueRef<Chain>(host_context_);
  auto* token = new AsyncToken(std::move(chain));
  // AsyncToken created with a reference count of 2 because it will be returned
  // to the `async.execute` caller and also will be later on emplaced by the
  // asynchronously executed task. If the caller immediately will drop its
  // reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  AddRef(token);
  return token;
}

void AsyncRuntime::EmplaceToken(AsyncToken* token) {
  token->GetAsyncValue()->SetStateConcrete();
  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(token);
}

void AsyncRuntime::AwaitToken(AsyncToken* token) {
  std::array<RCReference<AsyncValue>, 1> ref{FormRef(token->GetAsyncValue())};
  host_context_->Await(ref);
}

AsyncGroup* AsyncRuntime::CreateGroup() { return new AsyncGroup(this); }

size_t AsyncRuntime::AddTokenToGroup(AsyncGroup* group, AsyncToken* token) {
  return group->AddToken(token);
}

void AsyncRuntime::AwaitGroup(AsyncGroup* group) {
  SmallVector<RCReference<AsyncValue>, 4> refs;
  refs.reserve(group->size());

  for (AsyncToken* token : group->GetAsyncTokens())
    refs.emplace_back(FormRef(token->GetAsyncValue()));

  host_context_->Await(refs);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
