/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

//===- dataset.h ------------------------------------------------*- C++ -*-===//
//
// This file declares abstract classes needed by the data pipline library.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_DATASET_H_
#define TFRT_LIB_DATA_DATASET_H_

#include <memory>

#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace data {

namespace internal {

template <typename SubClass>
void DestroyImpl(SubClass* ptr, HostAllocator* allocator) {
  ptr->~SubClass();
  allocator->DeallocateBytes(ptr, sizeof(SubClass));
}

template <typename... T>
bool IsEmpty(AsyncValueRef<llvm::Optional<std::tuple<T...>>> value) {
  return value.IsAvailable() && !value.hasValue();
}

template <typename... T, size_t... I>
static void AllocateTupleResult(
    MutableArrayRef<RCReference<AsyncValue>> results,
    AsyncValueRef<std::tuple<T...>>& t, HostContext* host,
    std::index_sequence<I...>) {
  std::ignore = std::initializer_list<int>{
      (results[I] = host->MakeUnconstructedAsyncValueRef<T>().ReleaseRCRef(),
       0)...};
}

template <typename... T, size_t... I>
static void EmplaceTupleResult(ArrayRef<AsyncValue*> results,
                               std::tuple<T...>& input,
                               std::index_sequence<I...>) {
  // Use braced-init-list to retrieve the results in the tuple in sequence.
  std::ignore = std::initializer_list<int>{
      (results[I]->emplace<std::decay_t<T>>(std::move(std::get<I>(input))),
       0)...};
}

}  // namespace internal

// We separate the IteratorBase from the templatized Iterator so that
// kernels can use IteratorBase::GetNextUntyped without being specialized for
// the output type.
class IteratorBase : public ReferenceCounted<IteratorBase> {
 public:
  explicit IteratorBase(HostContext* host) : host_(host) {}

  virtual ~IteratorBase() {}

  // Returns a vector of (N + 1) AsyncValues where N represents the number of
  // output value types of the child Iterator class. The first N AsyncValues
  // represent the decoupled values of the std::tuple<...> returned by the
  // GetNext(...). The last AsyncValue has a bool value which is true iff
  // the iterator has not reached end prior to this call.
  virtual SmallVector<RCReference<AsyncValue>, 4> GetNextUntyped(
      const ExecutionContext& exec_ctx) = 0;

 protected:
  // For access to Destroy().
  friend class ReferenceCounted<IteratorBase>;
  virtual void Destroy() = 0;

  // TODO(b/154971099): Remove this after the ExecutionContext change is
  // submitted.
  HostContext* host_;
};

template <typename... T>
class Iterator : public IteratorBase {
 public:
  explicit Iterator(HostContext* host) : IteratorBase(host) {}

  // If the iterator has reached end, returns an empty AsyncValueRef. Otherwise,
  // returns the AsyncValueRef of the next element and advances the iterator.
  virtual AsyncValueRef<std::tuple<T...>> GetNext(
      const ExecutionContext& exec_ctx) = 0;

  SmallVector<RCReference<AsyncValue>, 4> GetNextUntyped(
      const ExecutionContext& exec_ctx) override;
};

// TODO(rachelim): Define `DatasetContext` and `IteratorContext` as a container
// for common arguments to dataset and iterator constructors respectively.
template <typename... T>
class Dataset : public ReferenceCounted<Dataset<T...>> {
 public:
  virtual ~Dataset() {}

  // Creates an iterator that points to the first element of the dataset.
  // The iterator should keep +1 reference to the parent_dataset.
  virtual RCReference<Iterator<T...>> MakeIterator() = 0;

 private:
  // For access to Destroy().
  friend class ReferenceCounted<Dataset<T...>>;

  virtual void Destroy() = 0;
};

template <typename... T>
SmallVector<RCReference<AsyncValue>, 4> Iterator<T...>::GetNextUntyped(
    const ExecutionContext& exec_ctx) {
  auto input = GetNext(exec_ctx);
  SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(sizeof...(T) + 1);

  internal::AllocateTupleResult(results, input, host_,
                                std::make_index_sequence<sizeof...(T)>{});
  results[sizeof...(T)] = host_->MakeUnconstructedAsyncValueRef<bool>();

  if (!input) {
    for (int i = 0; i < sizeof...(T); i++) {
      results[i]->SetError(DecodedDiagnostic{"iterator reached end"});
    }
    results[sizeof...(T)]->emplace<bool>(false);
    return results;
  }

  input.AndThen([results = RCArray<AsyncValue>(results),
                 input = input.CopyRef()]() mutable {
    if (input.IsError()) {
      for (int i = 0; i < sizeof...(T); i++) {
        results[i]->SetError(input.GetError());
      }
      results[sizeof...(T)]->emplace<bool>(true);
      return;
    }
    // IDEA(donglin): We can optimize performance by constructing a list of
    // views of AsyncValue from AsyncValueRef<std::tuple<T...>> without moving
    // data.
    internal::EmplaceTupleResult(results.values(), input.get(),
                                 std::make_index_sequence<sizeof...(T)>{});
    results[sizeof...(T)]->emplace<bool>(true);
  });

  return results;
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_DATASET_H_
