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

//===- filter_dataset.h -----------------------------------------*- C++ -*-===//
//
// This file declares FilterDataset class which wraps around another Dataset
// instance and outputs those elements from the underlying dataset which satisfy
// a user-defined filter function.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_DATA_FILTER_DATASET_H_
#define TFRT_LIB_DATA_FILTER_DATASET_H_

#include <queue>

#include "dataset.h"
#include "map_dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace data {

template <typename... T>
class FilterDatasetIterator;

// FilterDataset takes elements from the underlying dataset and outputs those
// elements which satisfy a user-defined filter function.
template <typename... T>
class FilterDataset : public Dataset {
 public:
  explicit FilterDataset(RCReference<Dataset> input_dataset,
                         RCReference<const Function> filter_fn,
                         HostContext* host)
      : input_dataset_(std::move(input_dataset)),
        host_(host),
        allocator_(host->allocator()),
        filter_fn_(std::move(filter_fn)) {}

  // This class is not copyable or movable.
  FilterDataset(const FilterDataset&) = delete;
  FilterDataset& operator=(const FilterDataset&) = delete;

  RCReference<Iterator> MakeIterator() override;

 private:
  // Allow iterator to rely on private data members of this dataset.
  friend class FilterDatasetIterator<T...>;

  void Destroy() override {
    internal::DestroyImpl<FilterDataset<T...>>(this, allocator_);
  }

  RCReference<Dataset> input_dataset_;
  HostContext* host_;
  HostAllocator* allocator_;
  // The function should take value from the `input_dataset_` as input and
  // then output a boolean value.
  RCReference<const Function> filter_fn_;
};

template <typename... T>
class FilterDatasetIterator : public Iterator {
 public:
  explicit FilterDatasetIterator(
      RCReference<FilterDataset<T...>> parent_dataset)
      : Iterator(),
        parent_dataset_(std::move(parent_dataset)),
        input_iterator_(parent_dataset_->input_dataset_->MakeIterator()),
        num_false_predicate_(0),
        token_owned_(false) {}

  IterationResult GetNext(const ExecutionContext& exec_ctx) override {
    auto* host = exec_ctx.host();

    llvm::SmallVector<RCReference<AsyncValue>, 4> result_values;
    result_values.resize(sizeof...(T));
    for (size_t i = 0; i < sizeof...(T); ++i) {
      result_values[i] = host->MakeIndirectAsyncValue();
    }
    auto result_eof = host->MakeUnconstructedAsyncValueRef<bool>();
    auto result = IterationResult::Pending(std::move(result_values),
                                           std::move(result_eof));
    {
      mutex_lock lock(mu_);
      output_buffer_.push(result.CopyRef());
    }
    MaybeScheduleBackgroundTask(exec_ctx, false, 0);
    return result;
  }

 private:
  // This class is not copyable or movable.
  FilterDatasetIterator(const FilterDatasetIterator&) = delete;
  FilterDatasetIterator& operator=(const FilterDatasetIterator&) = delete;

  void Destroy() override {
    internal::DestroyImpl<FilterDatasetIterator>(this,
                                                 parent_dataset_->allocator_);
  }

  // This method ensures that the control flow (e.g. get value from the
  // input_iterator_) is only executed by the thread that holds the token. It
  // helps ensure in-order delivery while still allowing an unblocking
  // GetNext(...) API.
  //
  // If there is no value in the `output_buffer_`, or if the caller can not get
  // the token to run the control flow logic, this method will not schedule any
  // background task. Otherwise, this method will schedule background task as
  // appropriate. This method will recursively call itself again when the first
  // value in the `input_and_predicate_buffer_` becomes available.
  void MaybeScheduleBackgroundTask(const ExecutionContext& exec_ctx,
                                   bool is_token_owner, int callback_count)
      TFRT_EXCLUDES(mu_);

  int OutputBufferSize() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    return output_buffer_.size();
  }

  IterationResult DequeueOutputBuffer() TFRT_EXCLUDES(mu_) {
    mutex_lock lock(mu_);
    auto value = std::move(output_buffer_.front());
    output_buffer_.pop();
    return value;
  }

  RCReference<FilterDataset<T...>> parent_dataset_;
  RCReference<Iterator> input_iterator_;
  mutex mu_;
  // A queue of AsyncValue pairs. The first value of each pair is a value from
  // the `input_iterator_`. The second value of each pair is the result of
  // applying `filter_fn_` to the first value.
  std::queue<std::pair<IterationResult, IterationResult>>
      input_and_predicate_buffer_;
  // A queue of AsyncValues that have already been returned to the GetNext(...)
  // caller.
  std::queue<IterationResult> output_buffer_ TFRT_GUARDED_BY(mu_);
  // The number of AsyncValues in the predicate_buffer_ whose predicate value is
  // false. `num_false_predicate_` might be negative if fetch_sub(...) is
  // invoked before fetch_add(...).
  std::atomic<int32_t> num_false_predicate_;
  // This is a unique logical token for this iterator instance. It effectively
  // acts as a lock to ensure in-order delivery of results by guaranteeing that
  // at most one thread can take the next value from the input_iterator_ and
  // update values in output_buffer_.
  //
  // The thread which changes token_owned from false to true "holds" the token,
  // and can pass it on to a thread that runs the callback it schedules. The
  // token is released when token_owned_ is changed from true to false.
  bool token_owned_ TFRT_GUARDED_BY(mu_);
};

template <typename... T>
RCReference<Iterator> FilterDataset<T...>::MakeIterator() {
  return TakeRef(host_->Construct<FilterDatasetIterator<T...>>(FormRef(this)));
}

template <typename... T>
void FilterDatasetIterator<T...>::MaybeScheduleBackgroundTask(
    const ExecutionContext& exec_ctx, bool is_token_owner, int callback_count) {
  {
    mutex_lock lock(mu_);
    // There is no more output value to update. Release the token if the caller
    // owns the token and then return.
    if (output_buffer_.empty()) {
      if (is_token_owner) {
        token_owned_ = false;
      }
      return;
    }
    // Return since the token is already owned by another thread.
    if (!is_token_owner && token_owned_) return;
    // Take the token if the thread does not already own the token.
    token_owned_ = true;
  }
  // Only the thread that owns the token can execute the code below. This
  // ensures in-order delivery since at most one thread can take value from the
  // input_iterator_ and update the output value in the output_buffer_.

  // Fetch enough number of values from the `input_iterator_` to 1) satisfy the
  // values newly added in the `output_buffer_` and 2) compensate for the values
  // in the `input_and_predicate_buffer_` whose predicate value evaluates to
  // false. And schedule tasks to run the filter_fn for newly fetched values in
  // parallel.
  auto host = exec_ctx.host();
  int input_fetch_num = OutputBufferSize() -
                        input_and_predicate_buffer_.size() +
                        std::max(num_false_predicate_.load(), 0);
  const Function* filter_fn = parent_dataset_->filter_fn_.get();
  auto additional_fn_args =
      RCArray<AsyncValue>(llvm::SmallVector<AsyncValue*, 1>());
  for (int i = 0; i < input_fetch_num; i++) {
    auto input = input_iterator_->GetNext(exec_ctx);
    auto predicate_values =
        EnqueueFunction(filter_fn, additional_fn_args.CopyRef(),
                        RCArray<AsyncValue>(input.values), exec_ctx);
    assert(predicate_values.size() == 1);

    predicate_values[0]->AndThen(
        [predicate_values = predicate_values[0].CopyRef(),
         iterator = FormRef(this)]() mutable {
          if (!predicate_values->IsError() && !predicate_values->get<bool>()) {
            iterator->num_false_predicate_.fetch_add(1);
          }
        });
    auto predicate = IterationResult::Pending(std::move(predicate_values),
                                              input.eof.CopyRef());
    input_and_predicate_buffer_.push(
        std::make_pair(std::move(input), std::move(predicate)));
  }
  // After the first value in the `input_and_predicate_buffer_` becomes
  // available, the token owner should update `output_buffer` as appropriate,
  // then call MaybeScheduleBackgroundTask() again to schedule more tasks if
  // there are still unfilled outputs.
  assert(!input_and_predicate_buffer_.empty() &&
         "input_and_predicate_buffer should not be empty");
  auto next = std::move(input_and_predicate_buffer_.front());
  input_and_predicate_buffer_.pop();
  auto input = std::move(next.first);
  auto predicate = std::move(next.second);

  SmallVector<AsyncValue*, 4> async_value_ptrs;
  for (auto& value : input.values) async_value_ptrs.push_back(value.get());
  async_value_ptrs.push_back(input.eof.GetAsyncValue());
  async_value_ptrs.push_back(predicate.values[0].get());
  async_value_ptrs.push_back(predicate.eof.GetAsyncValue());
  host->RunWhenReady(async_value_ptrs, [exec_ctx, host, callback_count,
                                        input = std::move(input),
                                        predicate = std::move(predicate),
                                        iterator = FormRef(this)]() mutable {
    auto predicate_value = std::move(predicate.values[0]);
    auto predicate_eof = std::move(predicate.eof);
    if (predicate_eof.IsError()) {
      auto output = iterator->DequeueOutputBuffer();
      for (auto& value : output.values) {
        value->SetError(predicate_eof.GetError());
      }
      output.eof.SetError(predicate_eof.GetError());
    } else if (predicate_eof.get()) {
      // The input_iterator_ has been exhausted. Note that predicate_eof and
      // input.eof should have the same value.
      auto error = host->MakeErrorAsyncValueRef("iterator reached end");
      while (iterator->OutputBufferSize() > 0) {
        auto output = iterator->DequeueOutputBuffer();
        for (auto& value : output.values) {
          value->SetError(error->GetError());
        }
        output.eof.emplace(true);
      }
    } else if (predicate_value->IsError()) {
      auto output = iterator->DequeueOutputBuffer();
      for (auto& value : output.values) {
        value->SetError(predicate_value->GetError());
      }
      output.eof.SetError(predicate_value->GetError());
    } else if (predicate_value->get<bool>()) {
      // The input satisfies the predicate.
      auto output = iterator->DequeueOutputBuffer();
      for (int i = 0; i < sizeof...(T); ++i) {
        auto* output_value = cast<IndirectAsyncValue>(output.values[i].get());
        output_value->ForwardTo(std::move(input.values[i]));
      }
      output.eof.emplace(false);
    } else {
      iterator->num_false_predicate_.fetch_sub(1);
    }
    // If there are too many recursive calls, the stack size limit will be
    // exceeded and it will cause segmentation fault. The maximum number of
    // recursive calls depend on the OS level stack size and the size of the
    // recursive function, which is hard to know for sure.
    // We need to balance between threadpool scheduling overhead and the risk of
    // hitting stack size limit when choosing the frequency of scheduling the
    // callback in the threadpool.
    if (callback_count >= MAX_RECURSIVE_CALLS) {
      host->EnqueueWork([exec_ctx, iterator = std::move(iterator)] {
        iterator->MaybeScheduleBackgroundTask(exec_ctx, true, 0);
      });
    } else {
      iterator->MaybeScheduleBackgroundTask(exec_ctx, true, callback_count + 1);
    }
  });
}

}  // namespace data
}  // namespace tfrt

#endif  // TFRT_LIB_DATA_FILTER_DATASET_H_
