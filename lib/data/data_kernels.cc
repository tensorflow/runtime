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
#include "interleave_dataset.h"
#include "map_dataset.h"
#include "prefetch_dataset.h"
#include "range_dataset.h"
#include "repeat_dataset.h"
#include "slice_dataset.h"
#include "tf_record_dataset.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// SliceDataset
//===----------------------------------------------------------------------===//

// Create a dataset with the values specified in the args.
template <typename T>
RCReference<SliceDataset<T>> MakeDatasetFromValues(Chain chain,
                                                   RemainingArguments args,
                                                   HostContext* host) {
  std::vector<T> vector;
  for (int i = 0, e = args.size(); i < e; i++) {
    vector.push_back(args[i]->get<T>());
  }
  return TakeRef(host->Construct<SliceDataset<T>>(std::move(vector), host));
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
    Chain chain, RemainingArguments args, HostContext* host) {
  std::vector<DenseHostTensor> vector;
  for (int i = 0, e = args.size(); i < e; i++) {
    vector.push_back(args[i]->get<DenseHostTensor>().CopyRef());
  }
  return TakeRef(
      host->Construct<SliceDataset<DenseHostTensor>>(std::move(vector), host));
}

//===----------------------------------------------------------------------===//
// RangeDataset
//===----------------------------------------------------------------------===//

// Create a dataset that yields the specified range.
template <typename T>
RCReference<RangeDataset<T>> MakeRangeDataset(T start, T stop, T step,
                                              HostContext* host) {
  assert(step != 0 && "step size cannot be 0");
  return TakeRef(host->Construct<RangeDataset<T>>(start, stop, step, host));
}

//===----------------------------------------------------------------------===//
// MapDataset
//===----------------------------------------------------------------------===//

// TODO(rachelim): Support variable number of arguments.
template <typename T, typename... U>
RCReference<MapDataset<std::tuple<T>, std::tuple<U...>>> MakeMapDataset(
    RCReference<Dataset<T>>* dataset, RemainingArguments args,
    Attribute<Function> fn, HostContext* host) {
  assert((args.size() + 1 == fn->argument_types().size()) &&
         "The function inputs do not match the dataset input types.");
  assert(fn->result_types().size() == sizeof...(U) &&
         "The function outputs do not match the dataset output types.");

  return TakeRef(host->Construct<MapDataset<std::tuple<T>, std::tuple<U...>>>(
      dataset->CopyRef(), RCArray<AsyncValue>(args.values()),
      FormRef(&fn.get()), host));
}

//===----------------------------------------------------------------------===//
// InterleaveDataset
//===----------------------------------------------------------------------===//

// TODO(rachelim): Support variable number of arguments.
template <typename T, typename... U>
RCReference<InterleaveDataset<std::tuple<T>, std::tuple<U...>>>
MakeInterleaveDataset(RCReference<Dataset<T>>* dataset, int64_t cycle_length,
                      int64_t block_length, Attribute<Function> fn,
                      HostContext* host) {
  assert(fn->argument_types().size() == 1 &&
         "Interleave only supports functions with unary inputs.");
  assert(
      fn->result_types().size() == 1 &&
      "Interleave expects only one function output, which must be a dataset.");

  return TakeRef(
      host->Construct<InterleaveDataset<std::tuple<T>, std::tuple<U...>>>(
          dataset->CopyRef(), cycle_length, block_length, FormRef(&fn.get()),
          host));
}

//===----------------------------------------------------------------------===//
// TFRecordDataset
//===----------------------------------------------------------------------===//

RCReference<TFRecordDataset> MakeTFRecordDataset(std::string path,
                                                 HostContext* host) {
  return TakeRef(host->Construct<TFRecordDataset>(std::move(path), host));
}

//===----------------------------------------------------------------------===//
// RepeatDataset
//===----------------------------------------------------------------------===//

template <typename... T>
RCReference<RepeatDataset<T...>> MakeRepeatDataset(
    RCReference<Dataset<T...>>* dataset, Attribute<int32_t> count,
    HostContext* host) {
  return TakeRef(host->Construct<RepeatDataset<T...>>(dataset->CopyRef(),
                                                      count.get(), host));
}

//===----------------------------------------------------------------------===//
// BatchDataset
//===----------------------------------------------------------------------===//

template <typename... T>
RCReference<BatchDataset<T...>> MakeBatchDataset(
    RCReference<Dataset<T...>>* dataset, Attribute<int32_t> batch_size,
    Attribute<bool> same_input_metadata, HostContext* host) {
  return TakeRef(host->Construct<BatchDataset<T...>>(
      dataset->CopyRef(), batch_size.get(), same_input_metadata.get(), host));
}

//===----------------------------------------------------------------------===//
// PrefetchDataset
//===----------------------------------------------------------------------===//

template <typename... T>
RCReference<PrefetchDataset<T...>> MakePrefetchDataset(
    RCReference<Dataset<T...>>* dataset, HostContext* host) {
  return TakeRef(host->Construct<PrefetchDataset<T...>>(
      dataset->CopyRef(), host->GetNumWorkerThreads(), host));
}

//===----------------------------------------------------------------------===//
// Generic input pipeline kernels
//===----------------------------------------------------------------------===//

// Create an iterator that points to the first element in the dataset.
template <typename... T>
RCReference<Iterator<T...>> MakeIteratorFromDataset(
    RCReference<Dataset<T...>>* dataset) {
  return (*dataset)->MakeIterator();
}

// Get the next element from the iterator and advance iterator.
// The returned AsyncValueRef will contain error if the iterator has reached
// end prior to the method invocation.
// IDEA(tf_runtime_team): it may be useful to return optional value and let
// caller handle EOF properly.
// TODO(b/155918211): Handle asynchrous EOF from the input_iterator_
template <typename... T>
AsyncValueRef<std::tuple<T...>> IteratorGetNext(
    RCReference<Iterator<T...>>* iterator, Chain chain,
    const ExecutionContext& exec_ctx) {
  auto input = (*iterator)->GetNext(exec_ctx);
  if (internal::IsConcreteAndEmpty(input)) {
    EmitErrorAsync(exec_ctx, "iterator reached end");
  }

  return std::move(input.values);
}

// Executes body_fn repeatedly until the iterator reaches end.
//
// Requirements:
// 1) The first value of 'args' should be an iterator. The size of 'args' should
// be one larger than the size of 'results'.
// 2) The outputs of the 'body_fn' should have exactly the same list of types as
// the values of 'args' (except the first value).
//
// How to determine the inputs/outputs of the 'body_fn' and the kernel:
// 1) The outputs of Iterator::GetNextUntyped(), combined with the values of
// 'args' (except the first value), will be filled in as the arguments of the
// 'body_fn' for the first iteration.
// 2) The outputs of Iterator::GetNextUntyped(), combined with the outputs of
// the 'body_fn', will be filled in as the arguments of the 'body_fn' for the
// next iteration.
// 3) The outputs of the 'body_fn' of the last iteration will be used as the
// outputs of the kernel.
static void EnumerateIterator(RemainingArguments args, RemainingResults results,
                              Attribute<Function> body_fn,
                              const ExecutionContext& exec_ctx) {
  assert(args.size() - 1 == results.size() &&
         "argument count should be one larger than the results count");

  const Function* body_fn_ptr = &body_fn.get();
  auto& iterator = args[0]->get<std::unique_ptr<IteratorBase>>();
  auto num_results = results.size();

  SmallVector<AsyncValue*, 8> fn_args;
  SmallVector<RCReference<AsyncValue>, 4> fn_results;
  fn_results.resize(num_results);

  for (int i = 0; i < num_results; i++) {
    fn_results[i] = FormRef(args[i + 1]);
  }

  // IDEA(donglin): We can optimize performance by avoiding making the chain of
  // AndThen() too long as we keep constructing the next iteration's async
  // values before the output async values from the current iteration are
  // available.
  while (true) {
    SmallVector<RCReference<AsyncValue>, 4> values_and_bool =
        iterator->GetNextUntyped(exec_ctx);
    auto& is_eof = values_and_bool[values_and_bool.size() - 1];
    if (is_eof->IsAvailable() && is_eof->get<bool>()) {
      break;
    }
    fn_args.resize(values_and_bool.size() - 1 + num_results);

    for (int i = 0; i < values_and_bool.size() - 1; i++) {
      fn_args[i] = values_and_bool[i].release();
    }
    // Move the results from the last iteration to the args of this iteration.
    for (int i = 0; i < num_results; i++) {
      fn_args[values_and_bool.size() - 1 + i] = fn_results[i].release();
    }

    body_fn_ptr->Execute(fn_args, fn_results, exec_ctx.host());
    for (int i = 0; i < fn_args.size(); i++) {
      fn_args[i]->DropRef();
    }

    if (auto cancel_av = exec_ctx.host()->GetCancelAsyncValue()) {
      // Cancellation detected. Set results to the cancel async value and
      // return.
      for (int i = 0; i < num_results; i++) {
        results[i] = FormRef(cancel_av);
      }
      return;
    }

    // If there is error, propagate it to results and return.
    for (int i = 0; i < num_results; i++) {
      if (fn_results[i]->IsError()) {
        for (int j = 0; j < num_results; j++) {
          results[j] = fn_results[i].CopyRef();
        }
        return;
      }
    }
  }

  for (int i = 0; i < num_results; i++) {
    results[i] = std::move(fn_results[i]);
  }
}

//===----------------------------------------------------------------------===//
// Kernel registrations
//===----------------------------------------------------------------------===//

template <typename... T>
static void RegisterIteratorKernelsForType(KernelRegistry* registry,
                                           const std::string& suffix) {
  registry->AddKernel("data.make_iterator_from_dataset." + suffix,
                      TFRT_KERNEL(MakeIteratorFromDataset<T...>));
  registry->AddKernel("data.iterator_get_next." + suffix,
                      TFRT_KERNEL(IteratorGetNext<T...>));
}

// This is the entrypoint to the library.
void RegisterDataKernels(KernelRegistry* registry) {
  RegisterIteratorKernelsForType<std::string>(registry, "str");
  RegisterIteratorKernelsForType<int32_t>(registry, "i32");
  RegisterIteratorKernelsForType<int64_t>(registry, "i64");
  RegisterIteratorKernelsForType<float>(registry, "f32");
  RegisterIteratorKernelsForType<DenseHostTensor>(registry, "tensor");
  RegisterIteratorKernelsForType<DenseHostTensor, DenseHostTensor>(
      registry, "tensor_and_tensor");
  RegisterIteratorKernelsForType<DenseHostTensor, int64_t>(registry,
                                                           "tensor_and_i64");
  RegisterIteratorKernelsForType<float, int32_t>(registry, "f32_and_i32");
  registry->AddKernel("data.enumerate.iterator",
                      TFRT_KERNEL(EnumerateIterator));

  registry->AddKernel("data.make_dataset_from_values.i32",
                      TFRT_KERNEL(MakeDatasetFromValues<int32_t>));
  registry->AddKernel("data.make_dataset_from_values.i64",
                      TFRT_KERNEL(MakeDatasetFromValues<int64_t>));
  registry->AddKernel("data.make_dataset_from_values.str",
                      TFRT_KERNEL(MakeDatasetFromValues<std::string>));
  registry->AddKernel("data.make_dataset_from_values.tensor",
                      TFRT_KERNEL(MakeDatasetFromValues<DenseHostTensor>));

  registry->AddKernel("data.range_dataset.i64",
                      TFRT_KERNEL(MakeRangeDataset<int64_t>));
  registry->AddKernel("data.range_dataset.i32",
                      TFRT_KERNEL(MakeRangeDataset<int32_t>));

  registry->AddKernel("data.tf_record_dataset",
                      TFRT_KERNEL(MakeTFRecordDataset));

  registry->AddKernel("data.map_dataset.i32.i32",
                      TFRT_KERNEL(MakeMapDataset<int32_t, int32_t>));
  registry->AddKernel("data.map_dataset.i32.f32",
                      TFRT_KERNEL(MakeMapDataset<int32_t, float>));
  registry->AddKernel("data.map_dataset.i64.i64",
                      TFRT_KERNEL(MakeMapDataset<int64_t, int64_t>));
  registry->AddKernel(
      "data.map_dataset.str.tensor",
      TFRT_KERNEL(MakeMapDataset<std::string, DenseHostTensor>));
  registry->AddKernel(
      "data.map_dataset.i64.tensor_and_i64",
      TFRT_KERNEL(MakeMapDataset<int64_t, DenseHostTensor, int64_t>));
  registry->AddKernel(
      "data.map_dataset.str.tensor_and_i64",
      TFRT_KERNEL(MakeMapDataset<std::string, DenseHostTensor, int64_t>));
  registry->AddKernel("data.map_dataset.i32.f32_and_i32",
                      TFRT_KERNEL(MakeMapDataset<int32_t, float, int32_t>));

  registry->AddKernel("data.interleave_dataset.i32.i32",
                      TFRT_KERNEL(MakeInterleaveDataset<int32_t, int32_t>));

  registry->AddKernel("data.batch_dataset.tensor",
                      TFRT_KERNEL(MakeBatchDataset<DenseHostTensor>));
  registry->AddKernel("data.batch_dataset.i32",
                      TFRT_KERNEL(MakeBatchDataset<int32_t>));
  registry->AddKernel("data.batch_dataset.i64",
                      TFRT_KERNEL(MakeBatchDataset<int64_t>));
  registry->AddKernel("data.batch_dataset.tensor_and_i64",
                      TFRT_KERNEL(MakeBatchDataset<DenseHostTensor, int64_t>));

  registry->AddKernel("data.repeat_dataset.i32",
                      TFRT_KERNEL(MakeRepeatDataset<int32_t>));
  registry->AddKernel("data.repeat_dataset.i64",
                      TFRT_KERNEL(MakeRepeatDataset<int64_t>));
  registry->AddKernel("data.repeat_dataset.str",
                      TFRT_KERNEL(MakeRepeatDataset<std::string>));

  registry->AddKernel(
      "data.prefetch_dataset.tensor_and_tensor",
      TFRT_KERNEL(MakePrefetchDataset<DenseHostTensor, DenseHostTensor>));
  registry->AddKernel(
      "data.prefetch_dataset.tensor_and_i64",
      TFRT_KERNEL(MakePrefetchDataset<DenseHostTensor, int64_t>));
}

}  // namespace data
}  // namespace tfrt
