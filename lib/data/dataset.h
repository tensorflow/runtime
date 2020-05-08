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

#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace data {

template <typename... T>
struct IterationResult {
  // Construct IterationResult with valid values and eof = false.
  static IterationResult Values(std::tuple<T...> v, HostContext* host) {
    return IterationResult<T...>(
        host->MakeAvailableAsyncValueRef<std::tuple<T...>>(std::move(v)),
        host->MakeAvailableAsyncValueRef<bool>(false));
  }

  // Construct IterationResult with eof = true. `values` will have error.
  static IterationResult Eof(HostContext* host) {
    return IterationResult<T...>(
        host->MakeErrorAsyncValueRef("iterator reached end"),
        host->MakeAvailableAsyncValueRef<bool>(true));
  }

  // Construct IterationResult with error in both `values` and `eof`.
  static IterationResult Error(RCReference<AsyncValue> error) {
    return IterationResult<T...>(
        AsyncValueRef<std::tuple<T...>>(error.CopyRef()),
        AsyncValueRef<bool>(error.CopyRef()));
  }

  // Construct IterationResult with possibly unavailable `values` and `eof`.
  static IterationResult Pending(AsyncValueRef<std::tuple<T...>> v,
                                 AsyncValueRef<bool> e) {
    return IterationResult<T...>(std::move(v), std::move(e));
  }

  IterationResult(AsyncValueRef<std::tuple<T...>> v, AsyncValueRef<bool> e)
      : values(std::move(v)), eof(std::move(e)) {}

  // If GetNext(...) failed to fetch values due to error, both `values` and
  // `eof` should have the error.
  // If GetNext(...) failed to fetch values due to end of iterator, `eof` should
  // be true and `values` should have error.
  // If GetNext(...) successfully fetched values, `eof` should be false.
  AsyncValueRef<std::tuple<T...>> values;
  AsyncValueRef<bool> eof;
};

namespace internal {

template <typename SubClass>
void DestroyImpl(SubClass* ptr, HostAllocator* allocator) {
  ptr->~SubClass();
  allocator->DeallocateBytes(ptr, sizeof(SubClass));
}

template <typename... T>
bool IsConcreteAndEmpty(const IterationResult<T...>& result) {
  return result.eof.IsConcrete() && result.eof.get();
}

template <typename... T, size_t... I>
static void AllocateTupleResult(
    MutableArrayRef<RCReference<AsyncValue>> results,
    const AsyncValueRef<std::tuple<T...>>& input, HostContext* host,
    std::index_sequence<I...>) {
  std::ignore = std::initializer_list<int>{
      (results[I] = host->MakeUnconstructedAsyncValueRef<T>().ReleaseRCRef(),
       0)...};
}

template <typename... T, size_t... I>
static void EmplaceTupleResult(ArrayRef<AsyncValue*> results,
                               std::tuple<T...> input,
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
  explicit IteratorBase() {}

  virtual ~IteratorBase() {}

  // Returns a vector of (N + 1) AsyncValues where N represents the number of
  // output value types of the child Iterator class. The first N AsyncValues
  // represent the decoupled values of the std::tuple<...> returned by the
  // GetNext(...). The last AsyncValue has a bool value which is true if and
  // only if the iterator was exhausted. When the last AsyncValue is true, the
  // first N AsyncValues will be ErrorAsyncValue.
  virtual SmallVector<RCReference<AsyncValue>, 4> GetNextUntyped(
      const ExecutionContext& exec_ctx) = 0;

 protected:
  // For access to Destroy().
  friend class ReferenceCounted<IteratorBase>;
  virtual void Destroy() = 0;
};

template <typename... T>
class Iterator : public IteratorBase {
 public:
  explicit Iterator() {}

  // If the iterator has reached end, returns an empty AsyncValueRef. Otherwise,
  // returns the AsyncValueRef of the next element and advances the iterator.
  virtual IterationResult<T...> GetNext(const ExecutionContext& exec_ctx) = 0;

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
  auto values = std::move(input.values);
  auto eof = std::move(input.eof);

  // Initialize results.
  SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(sizeof...(T) + 1);
  internal::AllocateTupleResult(results, values, exec_ctx.host(),
                                std::make_index_sequence<sizeof...(T)>{});
  results[sizeof...(T)] =
      exec_ctx.host()->MakeUnconstructedAsyncValueRef<bool>();

  SmallVector<AsyncValue*, 4> async_value_ptrs;
  async_value_ptrs.push_back(values.GetAsyncValue());
  async_value_ptrs.push_back(eof.GetAsyncValue());
  exec_ctx.host()->RunWhenReady(
      async_value_ptrs,
      [results = RCArray<AsyncValue>(results), eof = std::move(eof),
       values = std::move(values)]() mutable {
        if (eof.IsError()) {
          for (int i = 0; i < sizeof...(T); i++) {
            results[i]->SetError(eof.GetError());
          }
          results[sizeof...(T)]->emplace<bool>(false);
          return;
        }
        if (eof.get()) {
          for (int i = 0; i < sizeof...(T); i++) {
            results[i]->SetError(DecodedDiagnostic{"iterator reached end"});
          }
          results[sizeof...(T)]->emplace<bool>(true);
          return;
        }
        // IDEA(donglin): We can optimize performance by constructing a list of
        // views of AsyncValue from AsyncValueRef<std::tuple<T...>> without
        // moving data.
        internal::EmplaceTupleResult(results.values(), std::move(values.get()),
                                     std::make_index_sequence<sizeof...(T)>{});
        results[sizeof...(T)]->emplace<bool>(false);
      });

  return results;
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_DATASET_H_
