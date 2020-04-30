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

//===- batch_dataset.h ------------------------------------------*- C++ -*-===//
//
// This file declares BatchDataset class which wraps around another Dataset
// instance and batches the underlying elements before returning them via
// GetNext().
//
// If the underlying dataset element type is a tensor, GetNext() should return a
// tensor with +1 dimension. If the underlying dataset element type is a scalar,
// GetNext() should return a 1-D tensor of the same scalar type.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_DATA_BATCH_DATASET_H_
#define TFRT_DATA_BATCH_DATASET_H_

#include "dataset.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/support/template_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace data {

template <typename... T>
class BatchDatasetIterator;

// Return typename Iterator<DenseHostTensor, ...> where DenseHostTensor is
// repeated sizeof...(T) times in the parameter pack.
template <size_t N>
using DHTIterator = RepeatTypeHelperT<Iterator, N, DenseHostTensor>;

// Return typename Dataset<DenseHostTensor, ...> where DenseHostTensor is
// repeated sizeof...(T) times in the parameter pack.
template <size_t N>
using DHTDataset = RepeatTypeHelperT<Dataset, N, DenseHostTensor>;

// Return typename std::tuple<DenseHostTensor, ...> where DenseHostTensor is
// repeated N times in the parameter pack.
template <size_t N>
using DHTTuple = RepeatTypeHelperT<std::tuple, N, DenseHostTensor>;

// Template for BatchDatasetHelper<T> where T is a scalar type. GetNext()
// should return a 1-D DenseHostTensor of the batched scalar values.
template <typename T>
llvm::Expected<DenseHostTensor> BatchValues(ArrayRef<T> values,
                                            HostAllocator* allocator) {
  static_assert(std::is_scalar<T>::value, "T needs to be a scalar type");
  // Construct a 1-D tensor as output.
  ssize_t size = values.size();
  TensorMetadata output_metadata(GetDType<T>(), {size});

  auto output_dht =
      DenseHostTensor::CreateUninitialized(output_metadata, allocator);
  if (!output_dht) {
    return MakeStringError("failed to create uninitialized tensor");
  }
  // Copy scalar values into the output tensor.
  MutableDHTArrayView<T> output_dht_view(output_dht.getPointer());
  std::copy(values.begin(), values.end(), output_dht_view.Elements().begin());

  return std::move(*output_dht);
}

// Template specialization for BatchDataset<DenseHostTensor>. GetNext() returns
// a DenseHostTensor with +1 dimension. The current implementation copies data
// of the batched tensors to construct a new tensor.
template <>
llvm::Expected<DenseHostTensor> BatchValues<DenseHostTensor>(
    ArrayRef<DenseHostTensor> values, HostAllocator* allocator);

template <std::size_t N, std::size_t Index, typename... T>
struct TuplesToDHTsHelper {
  // This method converts std::vector<std::tuple<T...>> into
  // std::tuple<DenseHostTensor...>. The i'th element of the output tuple is a
  // DenseHostTensor presenting batched value of i'th values of every input
  // tuple.
  static llvm::Expected<DHTTuple<N - Index>> Convert(
      std::vector<std::tuple<T...>>& tuples, HostAllocator* allocator) {
    using ElementT = typename std::tuple_element<Index, std::tuple<T...>>::type;
    // Move 'Index'th value of every tuple into a vector.
    std::vector<ElementT> values;
    for (int i = 0, e = tuples.size(); i < e; i++) {
      values.push_back(std::move(std::get<Index>(tuples[i])));
    }
    // Batch the vector of values into std::tuple<DenseHostTensor>.
    auto dht = BatchValues<ElementT>(ArrayRef<ElementT>(values), allocator);
    if (!dht) {
      return dht.takeError();
    }
    auto head_tuple = std::make_tuple(std::move(*dht));
    // Recursively convert the remaining values of tuples into
    // std::tuple<DenseHostTensor...>.
    auto tail_tuple =
        TuplesToDHTsHelper<N, Index + 1U, T...>::Convert(tuples, allocator);
    if (!tail_tuple) {
      return tail_tuple.takeError();
    }
    // Concatenate the first_tuple with the tail_tuple.
    return std::tuple_cat(std::move(head_tuple), std::move(*tail_tuple));
  }
};

// Base case for TuplesToDHTsHelper<N, Index, T...> where Index == N.
template <std::size_t N, typename... T>
struct TuplesToDHTsHelper<N, N, T...> {
  static llvm::Expected<std::tuple<>> Convert(
      std::vector<std::tuple<T...>>& tuples, HostAllocator* allocator) {
    return std::make_tuple();
  }
};

// Convert std::vector<std::tuple<T...>> to std::tuple<DenseHostTensor...>
template <typename... T>
static llvm::Expected<DHTTuple<sizeof...(T)>> TuplesToDHTs(
    std::vector<std::tuple<T...>>& tuples, HostAllocator* allocator) {
  return TuplesToDHTsHelper<sizeof...(T), 0, T...>::Convert(tuples, allocator);
}

template <typename... T>
class BatchDataset : public DHTDataset<sizeof...(T)> {
 public:
  explicit BatchDataset(RCReference<Dataset<T...>> input_dataset,
                        int32_t batch_size, HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        batch_size_(batch_size),
        host_(host),
        allocator_(host->allocator()) {}

  // This class is not copyable or movable.
  BatchDataset(const BatchDataset&) = delete;
  BatchDataset& operator=(const BatchDataset&) = delete;

  RCReference<DHTIterator<sizeof...(T)>> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class BatchDatasetIterator<T...>;

  void Destroy() override {
    internal::DestroyImpl<BatchDataset<T...>>(this, allocator_);
  }

  RCReference<Dataset<T...>> input_dataset_;
  int32_t batch_size_;
  HostContext* host_;
  HostAllocator* allocator_;
};

template <typename... T>
class BatchDatasetIterator : public DHTIterator<sizeof...(T)> {
 public:
  explicit BatchDatasetIterator(RCReference<BatchDataset<T...>> parent_dataset)
      : DHTIterator<sizeof...(T)>(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()) {}

  // This class is not copyable or movable.
  BatchDatasetIterator(const BatchDatasetIterator&) = delete;
  BatchDatasetIterator& operator=(const BatchDatasetIterator&) = delete;

  AsyncValueRef<DHTTuple<sizeof...(T)>> GetNext(
      const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<BatchDatasetIterator>(this,
                                                parent_dataset_->allocator_);
  }

  RCReference<BatchDataset<T...>> parent_dataset_;
  RCReference<Iterator<T...>> input_iterator_;
};

template <typename... T>
RCReference<DHTIterator<sizeof...(T)>> BatchDataset<T...>::MakeIterator() {
  return TakeRef(host_->Construct<BatchDatasetIterator<T...>>(FormRef(this)));
}

template <typename... T>
AsyncValueRef<DHTTuple<sizeof...(T)>> BatchDatasetIterator<T...>::GetNext(
    const ExecutionContext& exec_ctx) {
  llvm::SmallVector<RCReference<AsyncValue>, 4> async_values;
  // Get up to batch_size values from the underlying iterator.
  for (int i = 0; i < parent_dataset_->batch_size_; i++) {
    auto async_value = input_iterator_->GetNext(exec_ctx);
    if (!async_value) {
      break;
    }
    if (async_value.IsError()) {
      return AsyncValueRef<DHTTuple<sizeof...(T)>>(async_value.ReleaseRCRef());
    }
    async_values.push_back(async_value.ReleaseRCRef());
  }
  if (async_values.empty()) {
    return AsyncValueRef<DHTTuple<sizeof...(T)>>();
  }

  SmallVector<AsyncValue*, 4> async_value_ptrs;
  // Translate RCReference<AsyncValue> to AsyncValue*.
  for (auto& async_value : async_values) {
    async_value_ptrs.push_back(async_value.get());
  }
  auto async_result =
      exec_ctx.host()
          ->template MakeUnconstructedAsyncValueRef<DHTTuple<sizeof...(T)>>();
  exec_ctx.host()->RunWhenReady(
      async_value_ptrs, [exec_ctx, async_values = std::move(async_values),
                         async_result = async_result.CopyRef(),
                         parent_dataset = parent_dataset_.CopyRef()] {
        std::vector<std::tuple<T...>> values;
        values.reserve(parent_dataset->batch_size_);
        for (auto& async_value : async_values) {
          if (async_value->IsError()) {
            async_result.SetError(async_value->GetError());
            return;
          }
          auto& value = async_value->get<std::tuple<T...>>();
          values.push_back(std::move(value));
        }
        auto result = TuplesToDHTs<T...>(values, parent_dataset->allocator_);
        if (!result) {
          async_result.SetError(EmitError(exec_ctx, result.takeError()));
          return;
        }
        async_result.emplace(std::move(*result));
      });

  return std::move(async_result);
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_BATCH_DATASET_H_
