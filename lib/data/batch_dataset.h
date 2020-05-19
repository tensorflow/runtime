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

template <typename T>
TensorMetadata GetMetadataFromValue(T& value) {
  return TensorMetadata(GetDType<T>(), {});
}

template <>
inline TensorMetadata GetMetadataFromValue<DenseHostTensor>(
    DenseHostTensor& value) {
  return value.metadata();
}

// Recursive base case
template <size_t N>
static void GetInputMetadataHelper(
    const SmallVector<RCReference<AsyncValue>, 4>& input,
    const SmallVector<AsyncValueRef<TensorMetadata>, 4>& results) {}

// For every component in input, copy its metadata into the corresponding index
// in results when it is available.
template <size_t N, typename T, typename... RemainingT>
static void GetInputMetadataHelper(
    const SmallVector<RCReference<AsyncValue>, 4>& input,
    const SmallVector<AsyncValueRef<TensorMetadata>, 4>& results) {
  auto index = N - (sizeof...(RemainingT) + 1);
  // Emplace index-th metadata
  input[index]->AndThen([component = input[index].CopyRef(),
                         result = results[index].CopyRef()]() {
    result.emplace(GetMetadataFromValue(component->get<T>()));
  });
  GetInputMetadataHelper<N, RemainingT...>(input, results);
}

template <typename... T>
SmallVector<AsyncValueRef<TensorMetadata>, 4> GetInputMetadata(
    const SmallVector<RCReference<AsyncValue>, 4>& input, HostContext* host) {
  SmallVector<AsyncValueRef<TensorMetadata>, 4> metadatas;
  metadatas.resize(sizeof...(T));
  for (size_t i = 0; i < sizeof...(T); ++i) {
    metadatas[i] = host->MakeUnconstructedAsyncValueRef<TensorMetadata>();
  }
  GetInputMetadataHelper<sizeof...(T), T...>(input, metadatas);

  return metadatas;
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

struct CounterAndError {
  explicit CounterAndError(uint32_t value) : counter(value) {}

  std::atomic<uint32_t> counter;
  std::atomic<AsyncValue*> error{nullptr};
};

// Copies value from `value` into the slice_index-th slice of `temp` when it is
// ready and decrements the counter or forwards an error to `counter_and_error`.
// Additionally, checks that the metadata of `value` matches
// `expected_metadata`. When `counter_and_error` reaches 0, which means that all
// slices have been copied, move `temp` to `result` and make it available.
template <typename T>
void CopySlice(RCReference<AsyncValue> value,
               AsyncValueRef<TensorMetadata> expected_metadata,
               AsyncValueRef<DenseHostTensor> temp,
               RCReference<AsyncValue> result,
               CounterAndError* counter_and_error, size_t slice_index) {
  // `temp` is an allocated DenseHostTensor.
  assert(temp.IsAvailable());
  value->AndThen([value = value.CopyRef(),
                  expected_metadata = std::move(expected_metadata),
                  temp = std::move(temp), result = std::move(result),
                  slice_index, counter_and_error]() mutable {
    if (value->IsError()) {
      AsyncValue* null_value = nullptr;
      AsyncValue* error_value = value.release();
      // Set error if it hasn't already been set.
      //
      // Use memory_order_release for the success case so that error_value is
      // visible to other threads when they load with memory_order_acquire. For
      // the failure case, we do not care about expected_value, so we can use
      // memory_order_relaxed.
      if (!counter_and_error->error.compare_exchange_strong(
              null_value, error_value, std::memory_order_release,
              std::memory_order_relaxed)) {
        error_value->DropRef();
      }
    } else {
      // Verify that the value's metadata equals the expected_metadata.
      // IDEA(donglin): Do this check only in DEBUG mode.
      assert(GetMetadataFromValue(value->get<T>()) == expected_metadata.get());
      CopyDataHelper<T>(&value->get<T>(), &temp.get(), slice_index);
    }

    auto remaining_slices = counter_and_error->counter.fetch_sub(1) - 1;
    if (remaining_slices == 0) {
      // Use memory_order_consume so that writes to this atomic variable from
      // other threads are visible to this thread.
      auto* error_value =
          counter_and_error->error.load(std::memory_order_consume);
      // Forward the error if any, otherwise move `temp` to `result`.
      if (error_value != nullptr) {
        result->SetError(error_value->GetError());
        error_value->DropRef();
      } else {
        result->emplace<DenseHostTensor>(std::move(temp.get()));
      }
      delete counter_and_error;
    }
  });
}

template <typename T>
void CopyComponent(SmallVector<RCReference<AsyncValue>, 4>&& batched_inputs,
                   AsyncValueRef<TensorMetadata> expected_metadata,
                   AsyncValueRef<DenseHostTensor> temp,
                   RCReference<AsyncValue> result) {
  temp.AndThen([batched_inputs = std::move(batched_inputs),
                expected_metadata = std::move(expected_metadata),
                temp = temp.CopyRef(), result = std::move(result)]() mutable {
    // If there was an error in tensor allocation, forward it to the result.
    if (temp.IsError()) {
      result->SetError(temp.GetError());
      return;
    }
    auto* counter_and_error = new CounterAndError(batched_inputs.size());
    // Otherwise, when each input is ready, copy it to `temp`. When all
    // inputs are copied, move `temp` to `result`.
    for (size_t i = 0; i < batched_inputs.size(); ++i) {
      CopySlice<T>(batched_inputs[i].CopyRef(), expected_metadata.CopyRef(),
                   temp.CopyRef(), result.CopyRef(), counter_and_error,
                   /*slice_index=*/i);
    }
  });
}

// Recursive base case.
template <size_t N>
void CopyToBatchHelper(
    SmallVector<SmallVector<RCReference<AsyncValue>, 4>, 4>&& inputs,
    SmallVector<AsyncValueRef<TensorMetadata>, 4>&& expected_metadata,
    SmallVector<AsyncValueRef<DenseHostTensor>, 4>&& temp_batched_values,
    const SmallVector<RCReference<AsyncValue>, 4>& results) {}

// Copy inputs to batch when they are ready. This function applies recursively
// to one component (with type T) at a time.
template <size_t N, typename T, typename... RemainingT>
void CopyToBatchHelper(
    SmallVector<SmallVector<RCReference<AsyncValue>, 4>, 4>&& inputs,
    SmallVector<AsyncValueRef<TensorMetadata>, 4>&& expected_metadata,
    SmallVector<AsyncValueRef<DenseHostTensor>, 4>&& temp_batched_values,
    const SmallVector<RCReference<AsyncValue>, 4>& results) {
  auto index = N - (sizeof...(RemainingT) + 1);

  SmallVector<RCReference<AsyncValue>, 4> batched_inputs;
  batched_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    batched_inputs.push_back(std::move(inputs[i][index]));
  }

  CopyComponent<T>(
      std::move(batched_inputs), std::move(expected_metadata[index]),
      std::move(temp_batched_values[index]), results[index].CopyRef());

  CopyToBatchHelper<N, RemainingT...>(std::move(inputs),
                                      std::move(expected_metadata),
                                      std::move(temp_batched_values), results);
}

template <typename... T>
void CopyToBatch(
    SmallVector<SmallVector<RCReference<AsyncValue>, 4>, 4>&& inputs,
    SmallVector<AsyncValueRef<TensorMetadata>, 4>&& expected_metadata,
    SmallVector<AsyncValueRef<DenseHostTensor>, 4>&& temp_batched_values,
    const SmallVector<RCReference<AsyncValue>, 4>& results) {
  CopyToBatchHelper<sizeof...(T), T...>(
      std::move(inputs), std::move(expected_metadata),
      std::move(temp_batched_values), results);
}

// For each component in the batch, when the metadata is available, allocate a
// DenseHostTensor with the corresponding batch shape and dtype.
static SmallVector<AsyncValueRef<DenseHostTensor>, 4> AllocateOutputTensors(
    const SmallVector<AsyncValueRef<TensorMetadata>, 4>& metadatas,
    size_t batch_size, const ExecutionContext& exec_ctx) {
  SmallVector<AsyncValueRef<DenseHostTensor>, 4> results;
  results.reserve(metadatas.size());
  for (size_t i = 0; i < metadatas.size(); ++i) {
    auto result =
        exec_ctx.host()->MakeUnconstructedAsyncValueRef<DenseHostTensor>();
    metadatas[i].AndThen([exec_ctx, batch_size,
                          metadata = metadatas[i].CopyRef(),
                          result = result.CopyRef()]() {
      if (metadata.IsError()) {
        result.SetError(metadata.GetError());
        return;
      }
      SmallVector<ssize_t, 4> output_dims;
      output_dims.resize(metadata->shape.GetRank() + 1);
      output_dims[0] = batch_size;
      for (size_t i = 0; i < output_dims.size() - 1; ++i) {
        output_dims[i + 1] = metadata->shape.GetDimensionSize(i);
      }
      TensorMetadata batched_metadata(metadata->dtype, output_dims);
      auto dht = DenseHostTensor::CreateUninitialized(batched_metadata,
                                                      exec_ctx.host());
      if (!dht) {
        result.SetError(
            EmitError(exec_ctx, "failed to create uninitialized tensor"));
        return;
      }
      result.emplace(std::move(*dht));
    });
    results.push_back(std::move(result));
  }

  return results;
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
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        is_initialized_(false) {}

  // This class is not copyable or movable.
  BatchDatasetIterator(const BatchDatasetIterator&) = delete;
  BatchDatasetIterator& operator=(const BatchDatasetIterator&) = delete;

  IterationResultUntyped GetNextUntyped(
      const ExecutionContext& exec_ctx) override;

 private:
  void Destroy() override {
    internal::DestroyImpl<BatchDatasetIterator>(this,
                                                parent_dataset_->allocator_);
  }

  RCReference<BatchDataset<T...>> parent_dataset_;
  RCReference<Iterator<T...>> input_iterator_;
  // input_metadata_ contains TensorMetadata from each of the components of the
  // first element from input_iterator_. When same_input_metadata_ is true, we
  // can use input_metadata_ to allocate output tensors before inputs are
  // available.
  SmallVector<AsyncValueRef<TensorMetadata>, 4> input_metadata_;
  bool is_initialized_;
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
IterationResultUntyped BatchDatasetIterator<T...>::GetNextUntyped(
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();
  SmallVector<SmallVector<RCReference<AsyncValue>, 4>, 4> inputs;
  // Get up to batch_size values from the underlying iterator.
  for (int i = 0; i < parent_dataset_->batch_size_; ++i) {
    auto input = input_iterator_->GetNextUntyped(exec_ctx);
    if (internal::IsConcreteAndEmpty(input)) {
      break;
    }
    inputs.push_back(std::move(input.values));
  }
  if (inputs.empty()) {
    return IterationResultUntyped::Eof(host, sizeof...(T));
  }

  SmallVector<AsyncValueRef<TensorMetadata>, 4> metadata;
  if (parent_dataset_->same_input_metadata_) {
    // If all input values have the same metadata, record the metadata of the
    // the first input and re-use it to allocate output tensor for every batch.
    // This allows us to allocate output tensor before input values are
    // available except for the first input.
    //
    // This improves L1/L2 cache affinity of the data copying from the input
    // values to the output tensor because the same thread that computes the
    // input value can copy this value to the output tensor.
    if (!is_initialized_) {
      input_metadata_ = GetInputMetadata<T...>(inputs[0], host);
      is_initialized_ = true;
    }
    metadata.reserve(input_metadata_.size());
    for (const auto& m : input_metadata_) {
      metadata.push_back(m.CopyRef());
    }
  } else {
    metadata = GetInputMetadata<T...>(inputs[0], host);
  }

  auto temp_batched_values =
      AllocateOutputTensors(metadata, inputs.size(), exec_ctx);

  SmallVector<RCReference<AsyncValue>, 4> results;
  results.reserve(sizeof...(T));
  for (size_t i = 0; i < sizeof...(T); ++i) {
    results.push_back(host->MakeUnconstructedAsyncValueRef<DenseHostTensor>());
  }

  CopyToBatch<T...>(std::move(inputs), std::move(metadata),
                    std::move(temp_batched_values), results);
  return IterationResultUntyped::Values(std::move(results), host);
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_DATA_BATCH_DATASET_H_
