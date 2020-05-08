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
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
namespace data {

template <typename... T>
class BatchDatasetIterator;

// Return typename IterationResult<DenseHostTensor, ...> where DenseHostTensor
// is repeated N times in the parameter pack.
template <size_t N>
using DHTIterationResult =
    RepeatTypeHelperT<IterationResult, N, DenseHostTensor>;

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

// Convert a vector of values into a tuple of the same values.
template <std::size_t N>
DHTTuple<N> MakeTuple(SmallVector<DenseHostTensor, 4>& values) {
  assert(values.size() >= N);

  auto head_tuple = std::make_tuple(std::move(values[values.size() - N]));
  auto tail_tuple = MakeTuple<N - 1U>(values);
  return std::tuple_cat(std::move(head_tuple), std::move(tail_tuple));
}

template <>
inline std::tuple<> MakeTuple<0>(SmallVector<DenseHostTensor, 4>& values) {
  return std::make_tuple();
}

template <typename T>
TensorMetadata GetMetadataFromValue(T& value) {
  return TensorMetadata(GetDType<T>(), {});
}

template <>
inline TensorMetadata GetMetadataFromValue<DenseHostTensor>(
    DenseHostTensor& value) {
  return value.metadata();
}

// Given a tuple of values, return an array where the i-th element in the array
// is the TensorMetadata of i-th element in the tuple.
template <typename... T, size_t... I>
static std::array<TensorMetadata, sizeof...(T)> GetMetadatasFromValuesHelper(
    std::tuple<T...>& values, std::index_sequence<I...>) {
  std::array<TensorMetadata, sizeof...(T)> tensor_metadatas;
  // Use std::initializer_list to call GetMetadataFromValue(...) for every
  // element in the tuple.
  std::ignore = std::initializer_list<int>{
      (tensor_metadatas[I] = GetMetadataFromValue(std::get<I>(values)), 0)...};

  return tensor_metadatas;
}

// Once `value` is available, update `metadata` so that its i-th element
// will be the TensorMetadata of the i-th element of `value`.
template <typename... T>
void UpdateMetadataFromValue(
    const AsyncValueRef<std::tuple<T...>>& value,
    const AsyncValueRef<std::array<TensorMetadata, sizeof...(T)>>& metadata) {
  value.AndThen([value = value.CopyRef(), metadata = metadata.CopyRef()] {
    if (value.IsError()) {
      metadata.SetError(value.GetError());
      return;
    }
    auto metadata_value = GetMetadatasFromValuesHelper(
        value.get(), std::make_index_sequence<sizeof...(T)>{});
    metadata.emplace(std::move(metadata_value));
  });
}

// Copy bytes of `src` to the index-th element of `dst`. This is useful to batch
// multiple scalar values into a DenseHostTenor.
template <typename T>
void CopyDataHelper(T* src, DenseHostTensor* dst, int index) {
  int data_size = sizeof(*src);
  char* dst_ptr = static_cast<char*>(dst->data()) + index * data_size;
  std::memcpy(dst_ptr, src, data_size);
}

// Copy bytes of `src` to the index-th element of `dst`. This is useful to batch
// multiple DenseHostTensors into a DenseHostTenor.
template <>
inline void CopyDataHelper<DenseHostTensor>(DenseHostTensor* src,
                                            DenseHostTensor* dst, int index) {
  int data_size = src->DataSizeInBytes();
  char* dst_ptr = static_cast<char*>(dst->data()) + index * data_size;
  std::memcpy(dst_ptr, src->data(), data_size);
}

// Given two tuples `src` and `dst` of the same length, for every offset `i` in
// the tuple, copy bytes of i-th element of `src` to the index-th element of
// the i-th element of `dst`.
template <typename... T, size_t... I>
void CopyData(AsyncValueRef<std::tuple<T...>> src,
              AsyncValueRef<DHTTuple<sizeof...(T)>> dst, int index,
              std::index_sequence<I...>) {
  // Use std::initializer_list to call CopyDataHelper(...) for every element
  // in the tuples.
  std::ignore = std::initializer_list<int>{
      (CopyDataHelper(&std::get<I>(src.get()), &std::get<I>(dst.get()), index),
       0)...};
}

struct CounterAndError {
  explicit CounterAndError(uint32_t value) : counter(value) {}

  std::atomic<uint32_t> counter;
  std::atomic<AsyncValue*> error{nullptr};
};

// Given two tuples `value` and `temp_batched_value` of the same length,
// once the `value` is available, for every offset `i` in the tuple, copy
// bytes of i-th element of `value` to the `input_index`-th element of the
// i-th element of `temp_batched_value`.
// Decrement the counter in `counter_and_error` after the copy completes. If the
// counter reaches zero, move `temp_batched_value` into `batched_value`.
//
// The metadata of `value` should equal `expected_metadata`.
template <typename... T>
void CopyInputToOutputBuffer(
    AsyncValueRef<std::tuple<T...>> value,
    AsyncValueRef<std::array<TensorMetadata, sizeof...(T)>> expected_metadata,
    AsyncValueRef<DHTTuple<sizeof...(T)>> temp_batched_value,
    AsyncValueRef<DHTTuple<sizeof...(T)>> batched_value, int input_index,
    CounterAndError* counter_and_error, HostContext* host) {
  assert(temp_batched_value.IsAvailable());
  value.AndThen([value = value.CopyRef(),
                 expected_metadata = std::move(expected_metadata),
                 temp_batched_value = std::move(temp_batched_value),
                 batched_value = std::move(batched_value), input_index,
                 counter_and_error, host]() mutable {
    if (value.IsError()) {
      AsyncValue* expected_value = nullptr;
      AsyncValue* error_value =
          value.IsError() ? value.release() : temp_batched_value.release();
      // Use memory_order_release for the success case so that error_value is
      // visible to other threads when they load with memory_order_acquire. For
      // the failure case, we do not care about expected_value, so we can use
      // memory_order_relaxed.
      if (!counter_and_error->error.compare_exchange_strong(
              expected_value, error_value, std::memory_order_release,
              std::memory_order_relaxed)) {
        error_value->DropRef();
      }
    } else {
      // Verify that the value's metadata equals the expected_metadata.
      // IDEA(donglin): Do this check only in DEBUG mode.
      auto metadata = host->template MakeUnconstructedAsyncValueRef<
          std::array<TensorMetadata, sizeof...(T)>>();
      UpdateMetadataFromValue<T...>(value, metadata);
      for (int i = 0; i < sizeof...(T); i++) {
        assert(metadata.get()[i] == expected_metadata.get()[i] &&
               "value's metadata should equal the expected metadata");
      }
      CopyData(std::move(value), temp_batched_value.CopyRef(), input_index,
               std::make_index_sequence<sizeof...(T)>{});
    }
    if (counter_and_error->counter.fetch_sub(1) == 1) {
      // Use memory_order_consume so that writes to this atomic variable from
      // other threads are visible to this thread.
      auto* error_value =
          counter_and_error->error.load(std::memory_order_consume);
      if (error_value != nullptr) {
        batched_value.SetError(error_value->GetError());
        error_value->DropRef();
      } else {
        batched_value.emplace(std::move(temp_batched_value.get()));
      }
      delete counter_and_error;
    }
  });
}

// Allocate a std::tuple<DenseHotTensor, ...>. The DType of the i-th element of
// the tuple should be same as the i-th element of `metadata`. The shape of the
// i-th element of the tuple should
// be `batch_size` X `shape_of_i_th_element_of_metadatas`.
template <typename... T>
AsyncValueRef<DHTTuple<sizeof...(T)>> AllocateOutputBuffer(
    AsyncValueRef<std::array<TensorMetadata, sizeof...(T)>> metadata,
    int32_t batch_size, const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  auto result =
      host->template MakeUnconstructedAsyncValueRef<DHTTuple<sizeof...(T)>>();
  metadata.AndThen([metadata = metadata.CopyRef(), result = result.CopyRef(),
                    batch_size, host, exec_ctx]() mutable {
    if (metadata.IsError()) {
      result.SetError(metadata.GetError());
      return;
    }
    SmallVector<DenseHostTensor, 4> tensors;
    for (auto& element_metadata : metadata.get()) {
      SmallVector<ssize_t, 4> output_dims;
      output_dims.resize(element_metadata.shape.GetRank() + 1);
      output_dims[0] = batch_size;
      for (int i = 1; i < output_dims.size(); ++i) {
        output_dims[i] = element_metadata.shape.GetDimensionSize(i - 1);
      }
      TensorMetadata batched_metadata(element_metadata.dtype, output_dims);
      auto dht = DenseHostTensor::CreateUninitialized(batched_metadata, host);
      if (!dht) {
        result.SetError(
            EmitError(exec_ctx, "failed to create uninitialized tensor"));
        return;
      }
      tensors.push_back(std::move(*dht));
    }
    result.emplace(MakeTuple<sizeof...(T)>(tensors));
  });
  return result;
}

// BatchDataset wraps around another Dataset instance and batches the underlying
// elements before returning them via GetNext().
//
// If the underlying dataset element type is a tensor, GetNext() should return a
// tensor with +1 dimension. If the underlying dataset element type is a scalar,
// GetNext() should return a 1-D tensor of the same scalar type.
template <typename... T>
class BatchDataset : public DHTDataset<sizeof...(T)> {
 public:
  // If `same_input_metadata` is true, all values from the `input_dataset`
  // must have the DType and TensorShape.
  explicit BatchDataset(RCReference<Dataset<T...>> input_dataset,
                        int32_t batch_size, bool same_input_metadata,
                        HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        batch_size_(batch_size),
        same_input_metadata_(same_input_metadata),
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
  const int32_t batch_size_;
  const bool same_input_metadata_;
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

  DHTIterationResult<sizeof...(T)> GetNext(
      const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<BatchDatasetIterator>(this,
                                                parent_dataset_->allocator_);
  }

  RCReference<BatchDataset<T...>> parent_dataset_;
  RCReference<Iterator<T...>> input_iterator_;
  // input_metadata_ is the TensorMetadata of the first value from the
  // input_iterator_. When same_input_metadata_ is true, we can use
  // input_metadata_ to allocate output tensors before inputs are available.
  AsyncValueRef<std::array<TensorMetadata, sizeof...(T)>> input_metadata_;
};

template <typename... T>
RCReference<DHTIterator<sizeof...(T)>> BatchDataset<T...>::MakeIterator() {
  return TakeRef(host_->Construct<BatchDatasetIterator<T...>>(FormRef(this)));
}

// TODO(b/155918211): Handle asynchrous EOF from the input_iterator_
// IDEA(donglin): Consider scheduling the batch operation to the background
// threadpool explicitly. This can prevent GetNext() from doing memory copy
// synchronously regardless of whether the input values are available.
template <typename... T>
DHTIterationResult<sizeof...(T)> BatchDatasetIterator<T...>::GetNext(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  llvm::SmallVector<AsyncValueRef<std::tuple<T...>>, 4> values;
  // Get up to batch_size values from the underlying iterator.
  for (int i = 0; i < parent_dataset_->batch_size_; ++i) {
    auto input = input_iterator_->GetNext(exec_ctx);
    if (internal::IsConcreteAndEmpty(input)) {
      break;
    }
    auto value = std::move(input.values);
    if (value.IsError()) {
      return DHTIterationResult<sizeof...(T)>::Error(value.ReleaseRCRef());
    }
    values.push_back(std::move(value));
  }
  if (values.empty()) {
    return DHTIterationResult<sizeof...(T)>::Eof(host);
  }

  AsyncValueRef<std::array<TensorMetadata, sizeof...(T)>> metadata;
  if (parent_dataset_->same_input_metadata_) {
    // If all input values have the same metadata, record the metadata of the
    // the first input and re-use it to allocate output tensor for every batch.
    // This allows us to allocate output tensor before input values are
    // available except for the first input.
    //
    // This improves L1/L2 cache affinity of the data copying from the input
    // values to the output tensor because the same thread that computes the
    // input value can copy this value to the output tensor.
    if (!input_metadata_) {
      input_metadata_ = host->template MakeUnconstructedAsyncValueRef<
          std::array<TensorMetadata, sizeof...(T)>>();
      UpdateMetadataFromValue<T...>(values[0], input_metadata_);
    }
    metadata = input_metadata_.CopyRef();
  } else {
    metadata = host->template MakeUnconstructedAsyncValueRef<
        std::array<TensorMetadata, sizeof...(T)>>();
    UpdateMetadataFromValue<T...>(values[0], metadata);
  }

  // Schedule tasks to run map function for each input tensor and copy the
  // function output to the temp_batched_value. After all data is copied to
  // temp_batched_value, move temp_batched_value to batched_value into
  // batched_value and make batched_value available.
  auto temp_batched_value =
      AllocateOutputBuffer<T...>(metadata.CopyRef(), values.size(), exec_ctx);
  auto batched_value =
      host->template MakeUnconstructedAsyncValueRef<DHTTuple<sizeof...(T)>>();

  // Schedule a task to copy data from values to the temp_batched_value.
  temp_batched_value.AndThen(
      [values = std::move(values), metadata = std::move(metadata),
       temp_batched_value = temp_batched_value.CopyRef(),
       batched_value = batched_value.CopyRef(), host]() mutable {
        if (temp_batched_value.IsError()) {
          batched_value.SetError(temp_batched_value.GetError());
          return;
        }
        // counter_and_error is used to keep track of 1) the count of remaining
        // values that have not been computed and 2) first error from any value.
        auto* counter_and_error = new CounterAndError(values.size());
        for (int idx = 0; idx < values.size(); ++idx) {
          CopyInputToOutputBuffer(std::move(values[idx]), metadata.CopyRef(),
                                  temp_batched_value.CopyRef(),
                                  batched_value.CopyRef(), idx,
                                  counter_and_error, host);
        }
      });
  return DHTIterationResult<sizeof...(T)>::Pending(
      std::move(batched_value), host->MakeAvailableAsyncValueRef<bool>(false));
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_BATCH_DATASET_H_
