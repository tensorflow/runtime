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

// MLIR Async Runtime implemented on top of TFRT HostContext and host
// concurrency primitives.

#include "tfrt/cpu/jit/async_runtime.h"

#include <cstddef>
#include <memory>
#include <type_traits>

#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/concurrent_vector.h"
#include "tfrt/support/ref_count.h"

// -------------------------------------------------------------------------- //
// Define AsyncToken and AsyncGroup in the mlir::runtime namespace to implement
// opaque structs defined in the MLIR Async Runtime API header file.
// -------------------------------------------------------------------------- //

namespace mlir {
namespace runtime {

using tfrt::AsyncValueRef;
using tfrt::HostBuffer;
using tfrt::HostContext;
using tfrt::MakeConstructedAsyncValueRef;
using tfrt::RCReference;
using tfrt::cpu::jit::AsyncRuntime;
using tfrt::cpu::jit::AsyncRuntimeObject;

class AsyncToken : public AsyncRuntimeObject {
 public:
  explicit AsyncToken(HostContext* host, unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        chain_(MakeConstructedAsyncValueRef<tfrt::Chain>(host)) {}

  tfrt::AsyncValue* GetAsyncValue() const { return chain_.GetAsyncValue(); }

 private:
  AsyncValueRef<tfrt::Chain> chain_;
};

class AsyncValue : public AsyncRuntimeObject {
 public:
  explicit AsyncValue(HostContext* host, size_t size, size_t alignment,
                      unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        storage_(Storage::CanStoreInline(size, alignment)
                     ? MakeConstructedAsyncValueRef<Storage>(host)
                     : MakeConstructedAsyncValueRef<Storage>(host, host, size,
                                                             alignment)) {}

  void* GetStorage() const {
    if (storage_->is_inline) return &storage_->inline_buffer;
    return storage_->host_buffer->data();
  }

  tfrt::AsyncValue* GetAsyncValue() const { return storage_.GetAsyncValue(); }

 private:
  // If the requested async value storage is small, use the inlined storage,
  // fallback on the HostBuffer if the requested storage size is large.
  struct Storage {
    static const int kSize = 128;  // enough to fit memref descriptor of rank 5
    static const int kAlign = alignof(std::max_align_t);

    Storage() : is_inline(true) {}
    Storage(HostContext* host, size_t size, size_t alignment)
        : is_inline(false),
          host_buffer(HostBuffer::CreateUninitialized(size, alignment,
                                                      host->allocator())
                          .release()) {}

    ~Storage() {
      if (!is_inline) host_buffer->DropRef();
    }

    static bool CanStoreInline(size_t size, size_t alignment) {
      assert(llvm::isPowerOf2_32(alignment));
      return size <= kSize && alignment <= kAlign;
    }

    bool is_inline;
    union {
      std::aligned_storage<kSize, kAlign>::type inline_buffer;
      tfrt::HostBuffer* host_buffer;
    };
  };

  AsyncValueRef<Storage> storage_;
};

class AsyncGroup : public AsyncRuntimeObject {
  using AsyncTokens = tfrt::ConcurrentVector<AsyncToken*>;

 public:
  explicit AsyncGroup(AsyncRuntime* runtime, unsigned ref_count = 1)
      : AsyncRuntimeObject(ref_count),
        runtime_(runtime),
        async_tokens_(std::make_unique<AsyncTokens>(/*initial_capacity=*/16)) {}

  ~AsyncGroup() override {
    for (auto* obj : async_tokens_->ToArrayRef()) runtime_->DropRef(obj);
  }

  size_t AddToken(AsyncToken* token) {
    AsyncRuntime::AddRef(token);  // keep token alive while *this is alive
    return async_tokens_->emplace_back(token);
  }

  size_t size() const { return async_tokens_->size(); }

  llvm::SmallVector<tfrt::AsyncValue*, 4> GetAsyncValues() const {
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

/*static*/ void* AsyncRuntime::GetStorage(Value* value) {
  return value->GetStorage();
}

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncRuntime::Value* value) {
  return value->GetAsyncValue();
}

/*static*/ AsyncValue* AsyncRuntime::GetAsyncValue(AsyncRuntime::Token* token) {
  return token->GetAsyncValue();
}

/*static*/ SmallVector<AsyncValue*, 4> AsyncRuntime::GetAsyncValues(
    AsyncRuntime::Group* group) {
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
    AsyncRuntime::Token* token) {
  return static_cast<AsyncRuntimeObject*>(token);
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncRuntime::Value* value) {
  return static_cast<AsyncRuntimeObject*>(value);
}

/*static*/ AsyncRuntimeObject* AsyncRuntime::ToAsyncRuntimeObject(
    AsyncRuntime::Group* group) {
  return static_cast<AsyncRuntimeObject*>(group);
}

AsyncRuntime::Token* AsyncRuntime::CreateToken() {
  // AsyncRuntime::Token created with a reference count of 2 because it will be
  // returned to the `async.execute` caller and also will be later on emplaced
  // by the asynchronously executed task. If the caller immediately will drop
  // its reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  return new AsyncRuntime::Token(host_context_, /*ref_count=*/2);
}

void AsyncRuntime::SetAvailable(AsyncRuntime::Token* token) {
  token->GetAsyncValue()->SetStateConcrete();
  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(token);
}

void AsyncRuntime::AwaitToken(AsyncRuntime::Token* token) {
  std::array<RCReference<AsyncValue>, 1> ref{FormRef(token->GetAsyncValue())};
  host_context_->Await(ref);
}

AsyncRuntime::Value* AsyncRuntime::CreateValue(size_t size, size_t alignment) {
  // AsyncRuntime::Value created with a reference count of 2 because it will be
  // returned to the `async.execute` caller and also will be later on emplaced
  // by the asynchronously executed task. If the caller immediately will drop
  // its reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  return new AsyncRuntime::Value(host_context_, size, alignment,
                                 /*ref_count=*/2);
}

void AsyncRuntime::SetAvailable(AsyncRuntime::Value* value) {
  value->GetAsyncValue()->SetStateConcrete();
  // Async values created with a ref count `2` to keep token alive until the
  // async task completes. Drop extra reference explicitly when token emplaced.
  DropRef(value);
}

void AsyncRuntime::AwaitValue(AsyncRuntime::Value* value) {
  std::array<RCReference<AsyncValue>, 1> ref{FormRef(value->GetAsyncValue())};
  host_context_->Await(ref);
}

AsyncRuntime::Group* AsyncRuntime::CreateGroup() {
  return new AsyncRuntime::Group(this);
}

size_t AsyncRuntime::AddTokenToGroup(AsyncRuntime::Group* group,
                                     AsyncRuntime::Token* token) {
  return group->AddToken(token);
}

void AsyncRuntime::AwaitGroup(AsyncRuntime::Group* group) {
  SmallVector<RCReference<AsyncValue>, 4> refs;
  refs.reserve(group->size());

  for (AsyncRuntime::Token* token : group->GetAsyncTokens())
    refs.emplace_back(FormRef(token->GetAsyncValue()));

  host_context_->Await(refs);
}

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt
