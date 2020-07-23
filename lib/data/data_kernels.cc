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

//===- data_kernels.cc ----------------------------------------------------===//
//
// This file implements data kernels.
//
//===----------------------------------------------------------------------===//

#include "batch_dataset.h"
#include "filter_dataset.h"
#include "interleave_dataset.h"
#include "map_dataset.h"
#include "memory_dataset.h"
#include "prefetch_dataset.h"
#include "range_dataset.h"
#include "repeat_dataset.h"
#include "slice_dataset.h"
#include "tf_record_dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/rc_array.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// SliceDataset
//===----------------------------------------------------------------------===//

// Create a dataset with the values specified in the args.
template <typename T>
RCReference<SliceDataset<T>> MakeDatasetFromValues(
    Chain chain, RemainingArguments args, const ExecutionContext& exec_ctx) {
  std::vector<T> vector;
  for (int i = 0, e = args.size(); i < e; i++) {
    vector.push_back(args[i]->get<T>());
  }
  return TakeRef(exec_ctx.host()->Construct<SliceDataset<T>>(std::move(vector),
                                                             exec_ctx.host()));
}

// Add template specialization for DenseHostTensor because DenseHostTensor does
// not have a copy constructor. This implementation passes DenseHostTensor by
// reference. Passing tensor by reference is more performant and it would be
// 'correct' if we pass value to this kernel as attribute.
//
// IDEA(donglin): It is assumed that the input values are no longer accessed
// outside the SliceDataset after they are used to create the SliceDataset. We
// can pass values as attribute when we support e.g. TensorAttribute in TFRT.
template <>
RCReference<SliceDataset<DenseHostTensor>> MakeDatasetFromValues(
    Chain chain, RemainingArguments args, const ExecutionContext& exec_ctx) {
  std::vector<DenseHostTensor> vector;
  for (int i = 0, e = args.size(); i < e; i++) {
    vector.push_back(args[i]->get<DenseHostTensor>().CopyRef());
  }
  return TakeRef(exec_ctx.host()->Construct<SliceDataset<DenseHostTensor>>(
      std::move(vector), exec_ctx.host()));
}

//===----------------------------------------------------------------------===//
// RangeDataset
//===----------------------------------------------------------------------===//

// Create a dataset that yields the specified range.
RCReference<RangeDataset> MakeRangeDataset(int64_t start, int64_t stop,
                                           int64_t step,
                                           Attribute<uint8_t> element_type,
                                           const ExecutionContext& exec_ctx) {
  assert(step != 0 && "step size cannot be 0");
  auto dtype = ConvertBEFDataTypeToTensorDType(
      static_cast<BEFDataType>(element_type.get()));
  return TakeRef(exec_ctx.host()->Construct<RangeDataset>(
      start, stop, step, dtype, exec_ctx.host()));
}

//===----------------------------------------------------------------------===//
// MapDataset
//===----------------------------------------------------------------------===//

RCReference<MapDataset> MakeMapDataset(RCReference<Dataset>* dataset,
                                       RemainingArguments args,
                                       Attribute<Function> fn,
                                       const ExecutionContext& exec_ctx) {
  return TakeRef(exec_ctx.host()->Construct<MapDataset>(
      dataset->CopyRef(), RCArray<AsyncValue>(args.values()),
      FormRef(&fn.get()), exec_ctx.host()));
}

//===----------------------------------------------------------------------===//
// FilterDataset
//===----------------------------------------------------------------------===//

RCReference<FilterDataset> MakeFilterDataset(RCReference<Dataset>* dataset,
                                             Attribute<int64_t> arity,
                                             Attribute<Function> fn,
                                             const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  return TakeRef(host->Construct<FilterDataset>(
      (*dataset).CopyRef(), FormRef(&fn.get()), arity.get(), host));
}

//===----------------------------------------------------------------------===//
// InterleaveDataset
//===----------------------------------------------------------------------===//

RCReference<InterleaveDataset> MakeInterleaveDataset(
    RCReference<Dataset>* dataset, int64_t cycle_length, int64_t block_length,
    Attribute<int64_t> arity, Attribute<Function> fn,
    const ExecutionContext& exec_ctx) {
  assert(
      fn->result_types().size() == 1 &&
      "Interleave expects only one function output, which must be a dataset.");

  return TakeRef(exec_ctx.host()->Construct<InterleaveDataset>(
      dataset->CopyRef(), cycle_length, block_length, FormRef(&fn.get()),
      arity.get(), exec_ctx.host()));
}

//===----------------------------------------------------------------------===//
// TFRecordDataset
//===----------------------------------------------------------------------===//

RCReference<TFRecordDataset> MakeTFRecordDataset(
    std::string path, const ExecutionContext& exec_ctx) {
  auto num_worker_threads = exec_ctx.host()->GetNumWorkerThreads();
  // Default buffer size to 256 KB.
  int64_t buffer_size = 256 * 1024;
  return TakeRef(exec_ctx.host()->Construct<TFRecordDataset>(
      std::move(path), buffer_size, num_worker_threads, exec_ctx.host()));
}

//===----------------------------------------------------------------------===//
// RepeatDataset
//===----------------------------------------------------------------------===//

RCReference<RepeatDataset> MakeRepeatDataset(RCReference<Dataset>* dataset,
                                             int64_t count,
                                             const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  return TakeRef(
      host->Construct<RepeatDataset>(dataset->CopyRef(), count, host));
}

//===----------------------------------------------------------------------===//
// MemoryDataset
//===----------------------------------------------------------------------===//

template <typename... T>
RCReference<MemoryDataset<T...>> MakeMemoryDataset(
    RCReference<Dataset>* dataset, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  return TakeRef(
      host->Construct<MemoryDataset<T...>>(dataset->CopyRef(), host));
}

//===----------------------------------------------------------------------===//
// BatchDataset
//===----------------------------------------------------------------------===//

template <typename... T>
RCReference<BatchDataset<T...>> MakeBatchDataset(
    RCReference<Dataset>* dataset, int64_t batch_size,
    Attribute<bool> same_input_metadata, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  return TakeRef(host->Construct<BatchDataset<T...>>(
      dataset->CopyRef(), batch_size, same_input_metadata.get(), host));
}

//===----------------------------------------------------------------------===//
// PrefetchDataset
//===----------------------------------------------------------------------===//

RCReference<PrefetchDataset> MakePrefetchDataset(
    RCReference<Dataset>* dataset, int64_t prefetch_num,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  if (prefetch_num == -1) {
    prefetch_num = host->GetNumWorkerThreads();
  }
  return TakeRef(
      host->Construct<PrefetchDataset>(dataset->CopyRef(), prefetch_num, host));
}

//===----------------------------------------------------------------------===//
// Generic input pipeline kernels
//===----------------------------------------------------------------------===//

// Create an iterator that points to the first element in the dataset.
RCReference<Iterator> MakeIteratorFromDataset(RCReference<Dataset>* dataset) {
  return (*dataset)->MakeIterator();
}

// Get the next element from the iterator and advance iterator.
// The returned AsyncValueRef will contain error if the iterator has reached
// end prior to the method invocation.
// IDEA(tf_runtime_team): it may be useful to return optional value and let
// caller handle EOF properly.
static void IteratorGetNext(RCReference<Iterator>* iterator, Chain chain_in,
                            Result<Chain> chain_out, RemainingResults results,
                            const ExecutionContext& exec_ctx) {
  auto input = (*iterator)->GetNext(exec_ctx);
  auto* eof = input.eof.GetAsyncValue();
  assert(results.size() == input.values.size());
  // Provide a fast path for the case where EOF=false is known synchronously.
  // We don't provide fast path for the case where EOF=true is known
  // synchronously since it is uncommon for caller to keep getting next after
  // the iterator has reached end.
  if (eof->IsConcrete() && !eof->get<bool>()) {
    for (size_t i = 0, e = results.size(); i < e; ++i) {
      results[i] = std::move(input.values[i]);
    }
    chain_out.Emplace();
    return;
  }
  SmallVector<RCReference<IndirectAsyncValue>, 4> result_refs;
  result_refs.reserve(results.size());
  for (size_t i = 0, e = results.size(); i < e; ++i) {
    auto result = results.AllocateIndirectResultAt(i);
    result_refs.push_back(std::move(result));
  }
  eof->AndThen([exec_ctx, result_refs = std::move(result_refs),
                input = std::move(input)]() mutable {
    if (!input.eof.IsError() && input.eof.get()) {
      auto err = EmitErrorAsync(exec_ctx, "iterator reached end");
      for (size_t i = 0, e = result_refs.size(); i < e; ++i) {
        result_refs[i]->SetError(err->GetError());
      }
      return;
    }

    for (size_t i = 0, e = result_refs.size(); i < e; ++i) {
      result_refs[i]->ForwardTo(std::move(input.values[i]));
    }
  });
  chain_out.Emplace();
}

namespace {
// EnumerateContext orchestrates asynchronous iterator enumeration. It is
// allocated on the heap, and captures the enumeration state that has to
// be alive while the input iterator enumeration is in progress.
struct EnumerateContext {
  EnumerateContext(const ExecutionContext& exec_ctx,
                   RCReference<const Function> body_fn,
                   RCReference<Iterator> iterator, RemainingArguments* args,
                   RemainingResults* results)
      : exec_ctx(exec_ctx),
        body_fn(std::move(body_fn)),
        iterator(std::move(iterator)),
        num_results(results->size()) {
    // Initialize function result with enumerate arguments.
    for (int i = 0; i < num_results; i++) {
      fn_results.push_back(FormRef((*args)[i + 1]));
    }
    // Allocate indirect results for enumerate outputs.
    for (int i = 0; i < num_results; ++i) {
      enumerate_results.push_back(results->AllocateIndirectResultAt(i));
    }
  }

  void ProcessInputs(std::unique_ptr<EnumerateContext> ctx,
                     IterationResult iteration_result) {
    // Forward error in the EOF flag to all results.
    if (iteration_result.eof.IsError()) {
      ForwardEofError(iteration_result.eof);
      return;
    }

    // Forward EOF singal to all results.
    if (*iteration_result.eof) {
      ForwardResults();
      return;
    }

    // If any of the iteration values has an error, it will be propagated to
    // enumerate function outputs, and will be forwarded to the results.

    const size_t num_iterator_values = iteration_result.values.size();

    // Invoke the enumerator function.
    SmallVector<AsyncValue*, 8> fn_args;
    fn_args.resize(num_iterator_values + num_results);

    // Function arguments corresponding to iterator values.
    for (int i = 0; i < num_iterator_values; i++) {
      fn_args[i] = iteration_result.values[i].get();
    }

    // Function arguments that passed from last iteration results.
    for (int i = 0; i < num_results; i++) {
      fn_args[num_iterator_values + i] = fn_results[i].release();
    }

    body_fn->Execute(exec_ctx, fn_args, fn_results);

    // DropRef only on arguments from the previous iteration.
    for (int i = 0; i < num_results; i++) {
      fn_args[num_iterator_values + i]->DropRef();
    }

    // Stop enumeration if receive cancellation.
    if (Cancelled()) return;

    // If there is an error, propagate it to the results and return.
    if (ForwardedErrorResults()) return;

    // Get next input values.
    auto next = iterator->GetNext(exec_ctx);
    exec_ctx.host()->RunWhenReady(
        next.AsyncValues(),
        [ctx = std::move(ctx), next = std::move(next)]() mutable {
          auto* ctx_ptr = ctx.get();
          ctx_ptr->ProcessInputs(std::move(ctx), std::move(next));
        });
  }

  // Forward EOF flag error to all enumerate results.
  void ForwardEofError(const AsyncValueRef<bool>& eof) {
    assert(eof.IsError());
    for (int i = 0; i < num_results; ++i) {
      enumerate_results[i]->ForwardTo(eof.CopyRef());
    }
  }

  bool Cancelled() {
    auto cancel_av = exec_ctx.GetCancelAsyncValue();
    if (!cancel_av) return false;

    // Cancellation detected. Set results to the cancel async value.
    for (int i = 0; i < num_results; i++)
      enumerate_results[i]->ForwardTo(FormRef(cancel_av));
    return true;
  }

  // If there is an error in any of the enumerator function results, propagate
  // it to results and return.
  bool ForwardedErrorResults() {
    for (int i = 0; i < num_results; i++) {
      if (fn_results[i]->IsError()) {
        for (int j = 0; j < num_results; j++) {
          enumerate_results[j]->ForwardTo(fn_results[i].CopyRef());
        }
        return true;
      }
    }
    return false;
  }

  // Forward the last function results to the enumerate results.
  void ForwardResults() {
    for (int i = 0; i < num_results; i++) {
      enumerate_results[i]->ForwardTo(std::move(fn_results[i]));
    }
  }

  ExecutionContext exec_ctx;
  RCReference<const Function> body_fn;
  RCReference<Iterator> iterator;

  const size_t num_results;

  // Last successfull function invocation results (or the enumerate arguments,
  // if the function was never invoked).
  SmallVector<RCReference<AsyncValue>, 4> fn_results;
  // Enumerate results that will be forwared to the last function invocation
  // results, when iterator will be exhausted.
  SmallVector<RCReference<IndirectAsyncValue>, 4> enumerate_results;
};
}  // namespace

// Executes body_fn repeatedly until the iterator reaches end.
//
// Requirements:
// 1) The first value of 'args' should be an iterator. The size of 'args' should
// be one larger than the size of 'results'.
// 2) The outputs of the 'body_fn' should have exactly the same list of types as
// the values of 'args' (except the first value).
//
// How to determine the inputs/outputs of the 'body_fn' and the kernel:
// 1) The outputs of Iterator::GetNext(), combined with the values of
// 'args' (except the first value), will be filled in as the arguments of the
// 'body_fn' for the first iteration.
// 2) The outputs of Iterator::GetNext(), combined with the outputs of
// the 'body_fn', will be filled in as the arguments of the 'body_fn' for the
// next iteration.
// 3) The outputs of the 'body_fn' of the last iteration will be used as the
// outputs of the kernel.
static void EnumerateIterator(RemainingArguments args, RemainingResults results,
                              Attribute<Function> body_fn,
                              const ExecutionContext& exec_ctx) {
  assert(args.size() - 1 == results.size() &&
         "argument count should be one larger than the results count");

  auto& iterator = args[0]->get<RCReference<Iterator>>();
  auto ctx = std::make_unique<EnumerateContext>(
      exec_ctx, FormRef(&body_fn.get()), iterator.CopyRef(), &args, &results);

  // Request the first input from the iterator.
  auto next = iterator->GetNext(exec_ctx);
  exec_ctx.host()->RunWhenReady(
      next.AsyncValues(),
      [ctx = std::move(ctx), next = next.CopyRef()]() mutable {
        auto* ctx_ptr = ctx.get();
        ctx_ptr->ProcessInputs(std::move(ctx), std::move(next));
      });
}

//===----------------------------------------------------------------------===//
// Kernel registrations
//===----------------------------------------------------------------------===//

// This is the entrypoint to the library.
void RegisterDataKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_data.make_iterator",
                      TFRT_KERNEL(MakeIteratorFromDataset));
  registry->AddKernel("tfrt_data.iterator_get_next",
                      TFRT_KERNEL(IteratorGetNext));
  registry->AddKernel("tfrt_data.enumerate.iterator",
                      TFRT_KERNEL(EnumerateIterator));

  // TODO(b/155892156): Remove type specialization on dataset kernels.
  registry->AddKernel("tfrt_data.make_dataset_from_values.i32",
                      TFRT_KERNEL(MakeDatasetFromValues<int32_t>));
  registry->AddKernel("tfrt_data.make_dataset_from_values.i64",
                      TFRT_KERNEL(MakeDatasetFromValues<int64_t>));
  registry->AddKernel("tfrt_data.make_dataset_from_values.str",
                      TFRT_KERNEL(MakeDatasetFromValues<std::string>));
  registry->AddKernel("tfrt_data.make_dataset_from_values.tensor",
                      TFRT_KERNEL(MakeDatasetFromValues<DenseHostTensor>));

  registry->AddKernel("tfrt_data.range_dataset", TFRT_KERNEL(MakeRangeDataset));

  registry->AddKernel("tfrt_data.batch_dataset.tensor",
                      TFRT_KERNEL(MakeBatchDataset<DenseHostTensor>));
  registry->AddKernel("tfrt_data.batch_dataset.i32",
                      TFRT_KERNEL(MakeBatchDataset<int32_t>));
  registry->AddKernel("tfrt_data.batch_dataset.i64",
                      TFRT_KERNEL(MakeBatchDataset<int64_t>));
  registry->AddKernel("tfrt_data.batch_dataset.tensor_and_i64",
                      TFRT_KERNEL(MakeBatchDataset<DenseHostTensor, int64_t>));
  registry->AddKernel("tfrt_data.batch_dataset.i64_and_i64",
                      TFRT_KERNEL(MakeBatchDataset<int64_t, int64_t>));

  registry->AddKernel("tfrt_data.memory_dataset.i64",
                      TFRT_KERNEL(MakeMemoryDataset<int64_t>));
  registry->AddKernel("tfrt_data.memory_dataset.str",
                      TFRT_KERNEL(MakeMemoryDataset<std::string>));

  registry->AddKernel("tfrt_data.filter_dataset",
                      TFRT_KERNEL(MakeFilterDataset));
  registry->AddKernel("tfrt_data.interleave_dataset",
                      TFRT_KERNEL(MakeInterleaveDataset));
  registry->AddKernel("tfrt_data.map_dataset", TFRT_KERNEL(MakeMapDataset));
  registry->AddKernel("tfrt_data.prefetch_dataset",
                      TFRT_KERNEL(MakePrefetchDataset));
  registry->AddKernel("tfrt_data.repeat_dataset",
                      TFRT_KERNEL(MakeRepeatDataset));
  registry->AddKernel("tfrt_data.tf_record_dataset",
                      TFRT_KERNEL(MakeTFRecordDataset));
}

}  // namespace data
}  // namespace tfrt
