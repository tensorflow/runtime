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
// This file declares abstract classes needed by the data pipeline library.
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

// TODO(b/155892156): This class shadows IterationResult as part of the
// transition into type-erased iteration results. Eventually, this will replace
// IterationResult.
struct IterationResultUntyped {
  IterationResultUntyped(llvm::SmallVector<RCReference<AsyncValue>, 4> v,
                         AsyncValueRef<bool> e)
      : values(std::move(v)), eof(std::move(e)) {}

  // When these AsyncValues resolve, there are three possible states:
  //  (A) End of iteration: `eof` will be true, `values` will contain error
  //      values.
  //  (B) Error: both `eof` and `values` will contain error values.
  //  (C) Success: `eof` should be false; `values` will contain the values from
  //      iteration.
  llvm::SmallVector<RCReference<AsyncValue>, 4> values;
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

bool IsConcreteAndEmpty(const IterationResultUntyped& result);

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

  virtual IterationResultUntyped GetNextUntyped(
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

  virtual IterationResult<T...> GetNext(const ExecutionContext& exec_ctx) = 0;

  IterationResultUntyped GetNextUntyped(
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
IterationResultUntyped Iterator<T...>::GetNextUntyped(
    const ExecutionContext& exec_ctx) {
  auto input = GetNext(exec_ctx);
  auto values = std::move(input.values);
  auto eof = std::move(input.eof);

  // Convert `values` to a vector of untyped RCReference<AsyncValue>s when
  // they are available.
  SmallVector<RCReference<AsyncValue>, 4> untyped_values;
  untyped_values.resize(sizeof...(T));
  internal::AllocateTupleResult(untyped_values, values, exec_ctx.host(),
                                std::make_index_sequence<sizeof...(T)>{});

  SmallVector<AsyncValue*, 4> async_value_ptrs;
  async_value_ptrs.push_back(values.GetAsyncValue());
  async_value_ptrs.push_back(eof.GetAsyncValue());
  exec_ctx.host()->RunWhenReady(
      async_value_ptrs,
      [untyped_values_ref = RCArray<AsyncValue>(untyped_values),
       values = values.CopyRef(), eof = eof.CopyRef()]() {
        // We can only unpack values if we know that eof is false and there is
        // no error.
        if (eof.IsError()) {
          for (size_t i = 0; i < sizeof...(T); ++i) {
            untyped_values_ref[i]->SetError(eof.GetError());
          }
          return;
        }
        if (eof.get()) {
          for (size_t i = 0; i < sizeof...(T); ++i) {
            untyped_values_ref[i]->SetError(
                DecodedDiagnostic{"iterator reached end"});
          }
          return;
        }
        // TODO(b/155918211): Currently, there's an issue in BatchDataset
        // where EOF is set to false, instead of error, when there is an error
        // in batching. Fixing that involves having batch dataset's EOF be
        // available only asynchronously, but that will break all downstream
        // datasets (for now) until we support async EOF everywhere.
        if (values.IsError()) {
          for (size_t i = 0; i < sizeof...(T); ++i) {
            untyped_values_ref[i]->SetError(values.GetError());
          }
          return;
        }
        internal::EmplaceTupleResult(untyped_values_ref.values(),
                                     std::move(values.get()),
                                     std::make_index_sequence<sizeof...(T)>{});
      });

  return {std::move(untyped_values), std::move(eof)};
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_DATASET_H_
