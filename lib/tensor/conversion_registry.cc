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

//===- conversion_registry.cc -----------------------------------*- c++ -*-===//
//
// This file implements Tensor Conversion Function and its registry.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/conversion_registry.h"

#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/support/logging.h"

namespace tfrt {

namespace {

struct TensorConversionFnRegistryContext : public SharedContext {
  explicit TensorConversionFnRegistryContext(HostContext* host) {}
  std::unique_ptr<TensorConversionFnRegistry> registry = nullptr;
};
}  // namespace

TensorFormats TensorFormats::Create(
    llvm::ArrayRef<Tensor::Subclass> sbuclasses) {
  uint32_t allowed_formats = 0;
  for (auto subclass : sbuclasses)
    allowed_formats |= 1 << static_cast<uint32_t>(subclass);
  return {allowed_formats};
}

bool TensorFormats::Contains(Tensor::Subclass subclass) const {
  return (allowed_formats & (1 << static_cast<uint32_t>(subclass)));
}

void TensorConversionFnRegistry::AddTensorConversionFn(ConversionKey key,
                                                       TensorConversionFn fn) {
  bool added = conversion_fn_map_.try_emplace(key, fn).second;
  (void)added;
  assert(added && "Re-registered existing TensorConversionFn");
}

TensorConversionFn TensorConversionFnRegistry::GetTensorConversionFn(
    ConversionKey key) const {
  auto it = conversion_fn_map_.find(key);
  return it == conversion_fn_map_.end() ? nullptr : it->second;
}

AsyncValueRef<Tensor> TransferTensorTo(const Tensor& tensor, const Device& src,
                                       const Device& dst,
                                       TensorFormats allowed_formats,
                                       HostContext* host) {
  auto& shared_ctx =
      host->GetOrCreateSharedContext<TensorConversionFnRegistryContext>();
  assert(shared_ctx.registry && "does not have a TensorConversionFnRegistry");
  auto conversion_fn = shared_ctx.registry->GetTensorConversionFn(
      {tensor.subclass(), &dst.type()});
  if (!conversion_fn) {
    return host->MakeErrorAsyncValueRef("cannot find conversion function.");
  }
  // TODO(fishx): Avoid constructing ExecutionContext here.
  auto req_ctx = RequestContext::Create(host, /*resource_context=*/nullptr);
  ExecutionContext exec_ctx(std::move(req_ctx));
  return conversion_fn(tensor, src, dst, allowed_formats, exec_ctx);
}

static std::vector<TensorConversionFnRegistration>*
GetStaticTensorConversionFnRegistrations() {
  static std::vector<TensorConversionFnRegistration>* ret =
      new std::vector<TensorConversionFnRegistration>;
  return ret;
}

void AddStaticTensorConversionFn(TensorConversionFnRegistration func) {
  GetStaticTensorConversionFnRegistrations()->push_back(func);
}

void RegisterTensorConversionFns(HostContext* host) {
  auto& shared_ctx =
      host->GetOrCreateSharedContext<TensorConversionFnRegistryContext>();
  assert(!shared_ctx.registry && "already have a TensorConversionFnRegistry");
  shared_ctx.registry = std::make_unique<TensorConversionFnRegistry>();

  for (auto func : *GetStaticTensorConversionFnRegistrations()) {
    func(shared_ctx.registry.get());
  }
}

}  // namespace tfrt
