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

//===- dispatch_utils.h -----------------------------------------*- C++ -*-===//
//
// This file contains op handler agnostic helpers for executing dispatching ops.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_CORE_RUNTIME_DISPATCH_UTILS_H_
#define TFRT_CORE_RUNTIME_DISPATCH_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/op_metadata_function.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {

// Executes `invocation` in a op_handler agnostic manner.  This is the public
// entry point into this header.
//
// Op handler specific behavior is injected into `ExecuteOnOpHandler` via
// `OpHandlerTraits`.  `OpHandlerTraits` needs to be a class with the following
// members:
//
// struct XXXOpHandlerTraits {
//   // The type of inputs accepted by the dispatch function for this
//   op_handler. using InputTensorTy = ...;
//
//   // The OpEntry for operations that run on this op_handler.
//   using OpEntryTy = ...;
//
//   // If this typedef is present, ExecuteOnOpHandler exposes an overload that
//   // takes a value of `OpHandlerInfoTy` and pipes in this value to the
//   // MaybeConvertTensor and Dispatch callbacks.
//   using OpHandlerInfoTy = ...;
//
//   // If `arg_tensor` needs to be convered to a op_handler specific format
//   then
//   // does so, stores the converted value in *converted and returns true.
//   // Otherwise returns false and leaves *converted untouched.
//   //
//   // The op_handler_info argument is required only if the `OpHandlerInfoTy`
//   typedef
//   // is present.
//   static bool MaybeConvertTensor(const OpEntryTy& op_entry,
//                                  [const OpHandlerInfoTy& op_handler_info,]
//                                  const Tensor& arg_tensor,
//                                  const ExecutionContext& exec_ctx,
//                                  RCReference<AsyncValue>* converted);
//
//   // Executes the op_handler specific dispatch function for the op
//   corresponding
//   // to `op_entry`.
//   //
//   // The op_handler_info argument is required only if the `OpHandlerInfoTy`
//   typedef
//   // is present.
//   static void Dispatch(const OpEntryTy& op_entry,
//                        [const OpHandlerInfoTy& op_handler_info,]
//                        ArrayRef<InputTensorTy*> inputs,
//                        const OpAttrsRef& attrs,
//                        ArrayRef<TensorMetadata> result_mds,
//                        MutableArrayRef<RCReference<AsyncValue>> results,
//                        AsyncValueRef<Chain>* chain,
//                        const ExecutionContext& exec_ctx);
// };
//
// This overload will be SFINAE'ed out if OpHandlerTraits::OpHandlerInfoTy
// doesn't exist.
template <typename OpHandlerTraits>
bool ExecuteOnOpHandler(
    bool update_chain, const OpInvocation& invocation,
    RCReference<Device> device, typename OpHandlerTraits::OpEntryTy op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info);

template <typename OpHandlerTraits>
bool ExecuteOnOpHandler(bool update_chain, const OpInvocation& invocation,
                        RCReference<Device> device,
                        typename OpHandlerTraits::OpEntryTy op_entry);

namespace internal {
// Internal implementaion details, please do not depend on things inside this
// namespace.

enum class MDFunctionExecResult {
  kMetadataUnavailable,
  kError,
  kSuccess,
};

// Executes `metadata_fn` for the op invocation designated by `invocation`.
//
// The three possible values of `MDFunctionExecResult` denote the three possible
// outcomes:
//
//  - kMetadataUnavailable: TensorMetadata corresponding to some inputs is
//    unavailable.  Nothing was done and the caller needs to deal with the
//    situation.
//
//  - kError: The metadata function execution resulted in an error, either
//    because one of the TensorMetadata input was in an error state or because
//    the metadata function resulted in an error.  An error has been propagated
//    to the chain and normal outputs.
//
//  - kSuccess: The metadata function was successful and the TensorMetadata for
//    the op results have been added to `result_mds`.
MDFunctionExecResult ExecuteMetadataFunction(
    const OpMetadataFn& metadata_fn, const OpInvocation& invocation,
    SmallVectorImpl<TensorMetadata>& result_mds);

using MetadataIsReadyCallback = llvm::unique_function<void(
    const ExecutionContext& exec_ctx, MutableArrayRef<TensorHandle> arguments,
    const OpAttrsRef& attrs, size_t num_results,
    const SmallVector<TensorMetadata, 4>& result_mds,
    SmallVectorImpl<AsyncValueRef<Tensor>>* result_tensor_avs,
    AsyncValueRef<Chain>* chain)>;

// Waits until all the TensorMetadata inputs for `invocation` are available and
// then:
//
//  - If any of the TensorMetadata inputs are found to be errors then propagates
//    errors to all outputs and returns.
//  - If all the TensorMetadata inputs are non-errors then invokes `callback`.
//
// If `update_chain` is true then the op is expected to return an out chain and
// `invocation.chain` will be updated to it. Otherwise this only reads
// `invocation.chain`.
void ExecuteWhenMetadataIsReady(const OpInvocation& invocation,
                                const OpMetadataFn& metadata_fn,
                                bool update_chain, RCReference<Device> device,
                                MetadataIsReadyCallback callback);

template <typename OpHandlerTraits>
class AsyncOpDispatcher {
 public:
  using InputTensorTy = typename OpHandlerTraits::InputTensorTy;

  explicit AsyncOpDispatcher(
      const ExecutionContext& exec_ctx, OpAttrsRef frozen_attrs,
      SmallVectorImpl<RCReference<AsyncValue>>&& arguments,
      AsyncValueRef<Chain> chain,
      const SmallVectorImpl<TensorMetadata>& result_mds,
      typename OpHandlerTraits::OpEntryTy op_entry,
      typename OpHandlerTraits::OpHandlerInfoTy op_handler_info)
      : exec_ctx_(exec_ctx),
        frozen_attrs_(std::move(frozen_attrs)),
        chain_(std::move(chain)),
        arguments_(std::move(arguments)),
        op_entry_(std::move(op_entry)),
        op_handler_info_(std::move(op_handler_info)) {
    result_mds_.reserve(result_mds.size());
    for (const TensorMetadata& md : result_mds) result_mds_.push_back(md);
  }

  void RunDispatchFunction();

  const OpAttrsRef& frozen_attrs() const { return frozen_attrs_; }

  const SmallVectorImpl<TensorMetadata>& result_mds() const {
    return result_mds_;
  }

  // The following accessors have a `_ref` suffix to indicate that they return a
  // mutable reference.

  SmallVector<RCReference<IndirectAsyncValue>, 4>& result_ind_avs_ref() {
    return result_ind_avs_;
  }

  SmallVector<AsyncValueRef<TensorMetadata>, 4>& result_missing_md_avs_ref() {
    return result_missing_md_avs_;
  }

  static void RunDispatchFunctionSync(
      typename OpHandlerTraits::OpEntryTy& op_entry,
      typename OpHandlerTraits::OpHandlerInfoTy op_handler_info,
      ArrayRef<RCReference<AsyncValue>> inputs, const OpAttrsRef& attrs,
      size_t num_results, ArrayRef<TensorMetadata> result_mds,
      MutableArrayRef<AsyncValueRef<TensorMetadata>> result_missing_md_avs,
      SmallVectorImpl<RCReference<AsyncValue>>* results,
      AsyncValueRef<Chain>* chain, const ExecutionContext& exec_ctx);

 private:
  void PropagateError(AsyncValue* error);

  bool MaybeConvertTensor(const Tensor& t, RCReference<AsyncValue>* converted) {
    return OpHandlerTraits::MaybeConvertTensor(op_entry_, op_handler_info_, t,
                                               exec_ctx_, converted);
  }

  template <typename InputTensorTy>
  static InputTensorTy* GetInputTensor(const RCReference<AsyncValue>& arg) {
    return GetInputTensorHelper(arg, TypeTag<InputTensorTy>());
  }

  static AsyncValue* GetInputTensorHelper(const RCReference<AsyncValue>& arg,
                                          TypeTag<AsyncValue>) {
    return arg.get();
  }

  template <typename InputTensorTy>
  static InputTensorTy* GetInputTensorHelper(const RCReference<AsyncValue>& arg,
                                             TypeTag<InputTensorTy>) {
    return &arg->get<InputTensorTy>();
  }

  ExecutionContext exec_ctx_;
  OpAttrsRef frozen_attrs_;
  SmallVector<TensorMetadata, 4> result_mds_;

  AsyncValueRef<Chain> chain_;
  SmallVector<RCReference<AsyncValue>, 4> arguments_;

  // These are the IndirectAsyncValue's for the results, that we need to
  // fulfill.
  SmallVector<RCReference<IndirectAsyncValue>, 4> result_ind_avs_;

  // If we had no metadata function, then these will be the AsyncValues that
  // need to be fulfilled with a TensorMetadata.
  SmallVector<AsyncValueRef<TensorMetadata>, 4> result_missing_md_avs_;

  typename OpHandlerTraits::OpEntryTy op_entry_;
  typename OpHandlerTraits::OpHandlerInfoTy op_handler_info_;
};

// If an error was detected on the input arguments for the op, then we propagate
// the error to the result AsyncValue's and do not run the dispatch function.
template <typename OpHandlerTraits>
LLVM_ATTRIBUTE_NOINLINE void AsyncOpDispatcher<OpHandlerTraits>::PropagateError(
    AsyncValue* error) {
  // Any unresolved tensor results become the error.
  for (auto& result : result_ind_avs_)
    if (result->IsUnresolvedIndirect()) result->ForwardTo(FormRef(error));

  // If the op lacked a shape function, then propagate the error into each
  // of the TensorHandle metadata results.
  for (auto& metadata_av : result_missing_md_avs_)
    if (metadata_av.IsUnavailable()) metadata_av.SetError(error->GetError());

  // If this is a side effecting operation, propagate the error through the
  // result.
  if (chain_ && chain_.IsUnavailable()) chain_.SetError(error->GetError());
}

template <typename OpHandlerTraits>
void AsyncOpDispatcher<OpHandlerTraits>::RunDispatchFunction() {
  // Get pointers to the InputTensorTy to pass into the dispatch function.  We
  // may discover on the fly that we need a conversion.  If so, handle that too.
  SmallVector<AsyncValue*, 4> async_args;
  for (auto& arg : arguments_) {
    // If any of the arguments ended up being an error, then propagate it and
    // bail out.
    if (arg->IsError()) return PropagateError(arg.get());

    auto& arg_tensor = arg->template get<Tensor>();
    RCReference<AsyncValue> copied_arg;

    // If we need an argument conversion, then do that now.
    if (MaybeConvertTensor(arg_tensor, &arg)) {
      // If any tensor conversion ended up being an error, then propagate it and
      // bail out.
      if (arg->IsError()) return PropagateError(arg.get());
      // If the argument conversion was async, then we have to wait for it.
      if (arg->IsUnavailable()) async_args.push_back(arg.get());
    }
  }

  // If any arguments required async conversions (e.g. copy off a op_handler),
  // then we have to wait for those arguments to complete.
  if (!async_args.empty()) {
    exec_ctx_.host()->RunWhenReady(
        async_args, [dispatch_info = std::move(*this)]() mutable {
          dispatch_info.RunDispatchFunction();
        });
    return;
  }

  // Finally, run the dispatch function.
  SmallVector<RCReference<AsyncValue>, 4> result_tensors;
  RunDispatchFunctionSync(op_entry_, op_handler_info_, arguments_,
                          frozen_attrs_, result_ind_avs_.size(), result_mds_,
                          result_missing_md_avs_, &result_tensors,
                          chain_ ? &chain_ : nullptr, exec_ctx_);

  // Fulfill the result async values with the results of the op.
  for (size_t i = 0, e = result_ind_avs_.size(); i != e; ++i) {
    auto& result_tensor = result_tensors[i];
    if (!result_tensor->IsError()) {
      if (result_ind_avs_[i])
        result_ind_avs_[i]->ForwardTo(std::move(result_tensor));
    } else {
      if (result_ind_avs_[i])
        result_ind_avs_[i]->SetError(result_tensor->GetError());
    }
  }
}

template <typename OpHandlerTraits>
/*static*/ void AsyncOpDispatcher<OpHandlerTraits>::RunDispatchFunctionSync(
    typename OpHandlerTraits::OpEntryTy& op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info,
    ArrayRef<RCReference<AsyncValue>> inputs, const OpAttrsRef& attrs,
    size_t num_results, ArrayRef<TensorMetadata> result_mds,
    MutableArrayRef<AsyncValueRef<TensorMetadata>> result_missing_md_avs,
    SmallVectorImpl<RCReference<AsyncValue>>* results,
    AsyncValueRef<Chain>* chain, const ExecutionContext& exec_ctx) {
  SmallVector<InputTensorTy*, 4> arg_tensors;
  arg_tensors.reserve(inputs.size());
  for (auto& arg : inputs) {
    arg_tensors.push_back(GetInputTensor<InputTensorTy>(arg));
  }

  results->resize(num_results);
  // Check if the host has been cancelled.
  if (auto* cancel_error = exec_ctx.GetCancelAsyncValue()) {
    // Any unresolved tensor results become the error.
    for (auto& result : *results) result = FormRef(cancel_error);

    // If the op lacked a shape function, then propagate the error into each
    // of the TensorHandle metadata results.
    for (auto& metadata_av : result_missing_md_avs)
      metadata_av.SetError(cancel_error->GetError());

    // If this is a side effecting operation, propagate the error through the
    // result.
    if (chain && *chain) chain->SetError(cancel_error->GetError());
    return;
  }
  // Finally, run the dispatch function.
  AsyncValueRef<Chain> op_chain;
  {
    TFRT_TRACE_SCOPE(tfrt::StrCat("RunDispatch: ", op_entry.op_name));

    OpHandlerTraits::Dispatch(op_entry, op_handler_info, arg_tensors, attrs,
                              result_mds, *results, &op_chain, exec_ctx);
  }
  if (chain && *chain) {
    assert(op_chain && "the op does not produce a required out chain.");
    op_chain.AndThen(
        [op_chain = op_chain.CopyRef(), chain = chain->CopyRef()]() {
          if (op_chain.IsError()) {
            chain.SetError(op_chain.GetError());
          } else {
            chain.emplace();
          }
        });
  }

  // result_missing_md_avs will be empty if there was a metadata function
  // which already computed these results.
  if (result_missing_md_avs.empty()) {
    return;
  }
  assert(result_missing_md_avs.size() == results->size());

  // Fulfill the result metadata async values with the results of the op.
  for (size_t i = 0, e = results->size(); i != e; ++i) {
    auto& result_tensor = (*results)[i];
    result_tensor->AndThen([md = result_missing_md_avs[i].CopyRef(),
                            result_tensor = result_tensor.get()]() mutable {
      if (result_tensor->IsError()) {
        md.SetError(result_tensor->GetError());
      } else {
        // Fulfill the metadata async_value.
        md.emplace(result_tensor->get<Tensor>().metadata());
      }
    });
  }
}

// Execute the dispatch function for an op when the metadata for the results is
// resolved (assuming there is a metadata function).  This fills in
// result_tensor_avs with AsyncValues that are the futures of the result of the
// op execution.  There are two primary cases to be aware of here - we could
// either have result metadata entries or not.
//
// We have result metadata's when there is a shape function for the op.  In that
// case, the result_md's ArrayRef specifies the results of the op, and
// result_md_avs is null.
//
// If there is no shape function, then result_mds is empty, and result_md_avs
// must be filled in with the AsyncValue's for the eventually computed shape
// results of the tensor op.
template <typename OpHandlerTraits>
void ExecuteWithResultMetadataResolved(
    const ExecutionContext& exec_ctx, MutableArrayRef<TensorHandle> arguments,
    const OpAttrsRef& attrs, size_t num_results,
    const SmallVector<TensorMetadata, 4>& result_mds,
    SmallVectorImpl<AsyncValueRef<TensorMetadata>>* result_md_avs,
    SmallVectorImpl<AsyncValueRef<Tensor>>* result_tensor_avs,
    AsyncValueRef<Chain>* chain, bool update_chain,
    typename OpHandlerTraits::OpEntryTy op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info) {
  // If we have no input metadatas (from a metadata function) then we need to
  // resolve the TensorHandle metadata's from the op results.
  if (result_md_avs) {
    result_md_avs->reserve(num_results);
    for (size_t i = 0; i != num_results; ++i) {
      result_md_avs->push_back(
          MakeUnconstructedAsyncValueRef<TensorMetadata>(exec_ctx.host()));
    }
  }

  // Keep track of all the non-resolved values to see if we can dispatch the
  // kernel immediately. If not we will "and then" on these non-resolved values.
  SmallVector<AsyncValue*, 4> async_args;
  async_args.reserve(arguments.size() + 1);
  SmallVector<RCReference<AsyncValue>, 4> arg_tensors;
  arg_tensors.reserve(arguments.size());

  assert((!update_chain || (chain && *chain)) &&
         "the op requires an in chain.");
  if (chain && *chain) {
    if (!chain->IsAvailable()) async_args.push_back(chain->GetAsyncValue());
    if (update_chain) {
      // TODO(fishx): Avoid this heap allocation.
      *chain = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
    }
  }

  for (auto& argument : arguments) {
    AsyncValue* async_tensor = argument.GetAsyncTensor();

    // Keep track of unavailable arguments so we can "and then" them.  We handle
    // errors through the slow path as well.
    if (!async_tensor->IsConcrete()) {
      async_args.push_back(async_tensor);
      arg_tensors.push_back(argument.ReleaseTensorRef());
      continue;
    }

    // If the argument is already available, then we can check to see if it is a
    // supported format, and issue a data copy eagerly if not.
    auto& arg_tensor = async_tensor->get<Tensor>();

    RCReference<AsyncValue> copy;
    if (OpHandlerTraits::MaybeConvertTensor(op_entry, op_handler_info,
                                            arg_tensor, exec_ctx, &copy)) {
      if (copy->IsError()) {
        // If any tensor conversion fails, set results to error.
        result_tensor_avs->reserve(num_results);
        for (size_t i = 0; i != num_results; ++i) {
          (*result_md_avs)[i].SetError(copy->GetError());
          auto diag_copy = copy->GetError();
          result_tensor_avs->push_back(
              MakeErrorAsyncValueRef(exec_ctx.host(), std::move(diag_copy)));
        }
        if (chain && *chain && update_chain) {
          chain->SetError(copy->GetError());
        }
        return;
      }
      // If the copy itself was async, then remember to "and then" it.
      if (copy->IsUnavailable()) async_args.push_back(copy.get());

      arg_tensors.push_back(std::move(copy));
    } else {
      arg_tensors.push_back(argument.ReleaseTensorRef());
    }
  }

  if (async_args.empty()) {
    // All input tensor and input chain are available. We can immediately
    // dispatch the kernel synchronously.
    SmallVector<RCReference<AsyncValue>, 4> result_tensors;
    SmallVector<AsyncValueRef<TensorMetadata>, 0> empty_md_avs;
    internal::AsyncOpDispatcher<OpHandlerTraits>::RunDispatchFunctionSync(
        op_entry, op_handler_info, arg_tensors, attrs, num_results, result_mds,
        result_md_avs ? *result_md_avs : empty_md_avs, &result_tensors,
        update_chain ? chain : nullptr, exec_ctx);
    result_tensor_avs->reserve(num_results);
    // Fulfill the result async values with the results of the op.
    for (size_t i = 0; i != num_results; ++i) {
      result_tensor_avs->push_back(
          AsyncValueRef<Tensor>(std::move(result_tensors[i])));
    }
    return;
  }

  // We have at least one async tensor input, so we need to run the
  // kernel when it resolves.
  internal::AsyncOpDispatcher<OpHandlerTraits> op_dispatcher(
      exec_ctx, attrs.freeze(), std::move(arg_tensors),
      update_chain ? chain->CopyRef() : AsyncValueRef<Chain>(), result_mds,
      std::move(op_entry), std::move(op_handler_info));

  // The results have to be immediately available, but we don't know what
  // concrete Tensor type they will be fulfilled with.  Create
  // IndirectAsyncValue's to handle this.
  result_tensor_avs->reserve(num_results);
  op_dispatcher.result_ind_avs_ref().reserve(num_results);
  for (size_t i = 0; i != num_results; ++i) {
    auto tensor = MakeIndirectAsyncValue(exec_ctx.host());
    op_dispatcher.result_ind_avs_ref().push_back(tensor.CopyRef());
    result_tensor_avs->push_back(AsyncValueRef<Tensor>(std::move(tensor)));
    if (result_md_avs) {
      op_dispatcher.result_missing_md_avs_ref().push_back(
          (*result_md_avs)[i].CopyRef());
    }
  }

  exec_ctx.host()->RunWhenReady(
      async_args, [op_dispatcher = std::move(op_dispatcher)]() mutable {
        op_dispatcher.RunDispatchFunction();
      });
}

// This is the slow-path that is run when it turns out that an input
// TensorHandle has an async metadata. This does not normally happen (normally
// the shapes are synchronously available but the data is async) but can happen
// when ops produce data dependent shapes or lack shape functions.
//
// Because this is a very slow path, we want it out of line.
template <typename OpHandlerTraits>
LLVM_ATTRIBUTE_NOINLINE void ExecuteWithMetadataAsync(
    const OpInvocation& invocation, bool update_chain,
    RCReference<Device> device, typename OpHandlerTraits::OpEntryTy op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info) {
  ExecuteWhenMetadataIsReady(
      invocation, op_entry.metadata_fn, update_chain, std::move(device),
      [op_entry = std::move(op_entry),
       op_handler_info = std::move(op_handler_info), update_chain](
          const ExecutionContext& exec_ctx,
          MutableArrayRef<TensorHandle> arguments, const OpAttrsRef& attrs,
          size_t num_results, const SmallVector<TensorMetadata, 4>& result_mds,
          SmallVectorImpl<AsyncValueRef<Tensor>>* result_tensor_avs,
          AsyncValueRef<Chain>* chain) mutable {
        ExecuteWithResultMetadataResolved<OpHandlerTraits>(
            exec_ctx, arguments, attrs, num_results, result_mds,
            /*result_md_avs=*/nullptr, result_tensor_avs, chain, update_chain,
            std::move(op_entry), std::move(op_handler_info));
      });
}

template <typename OpHandlerTraits>
bool ExecuteOnOpHandlerImpl(
    bool update_chain, const OpInvocation& invocation,
    RCReference<Device> device, typename OpHandlerTraits::OpEntryTy op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info) {
  using internal::ExecuteMetadataFunction;
  using internal::MDFunctionExecResult;

  // If this operation is unknown by this op_handler, then we fail to execute
  // without emitting an error.
  if (!op_entry.dispatch_fn) return false;

  // This gets filled in with the TensorMetadata's for the op results if it has
  // a registered metadata function.
  SmallVector<TensorMetadata, 4> result_mds;

  // If the op has a metadata function, then we make sure to execute it, because
  // it may be checking the op invariants, and the op implementation may be slow
  // or async - but we want to propagate the shape synchronously whenever
  // possible.
  if (op_entry.metadata_fn) {
    auto md_exec_result =
        ExecuteMetadataFunction(op_entry.metadata_fn, invocation, result_mds);
    if (md_exec_result == MDFunctionExecResult::kError) {
      return true;
    }

    if (md_exec_result == MDFunctionExecResult::kMetadataUnavailable) {
      internal::ExecuteWithMetadataAsync<OpHandlerTraits>(
          invocation, update_chain, std::move(device), std::move(op_entry),
          std::move(op_handler_info));
      return true;
    }
  }

  // Okay, now that the metadata function returned successfully, we can run our
  // op.

  SmallVector<AsyncValueRef<TensorMetadata>, 8> result_md_avs;
  SmallVector<AsyncValueRef<Tensor>, 8> result_tensor_avs;

  // Don't pass a pointer to result_md_avs if we had a metadata function.
  auto* result_md_avs_ptr = op_entry.metadata_fn ? nullptr : &result_md_avs;
  auto results = invocation.results;

  internal::ExecuteWithResultMetadataResolved<OpHandlerTraits>(
      invocation.exec_ctx, invocation.arguments, invocation.attrs,
      results.size(), result_mds, result_md_avs_ptr, &result_tensor_avs,
      invocation.chain, update_chain, std::move(op_entry),
      std::move(op_handler_info));

  for (size_t i = 0, e = results.size(); i != e; ++i) {
    if (op_entry.metadata_fn) {
      results[i] = TensorHandle(device.CopyRef(), result_mds[i],
                                std::move(result_tensor_avs[i]));
    } else {
      results[i] = TensorHandle(device.CopyRef(), std::move(result_md_avs[i]),
                                std::move(result_tensor_avs[i]));
    }
  }

  return true;
}
}  // namespace internal

template <typename OpHandlerTraits>
bool ExecuteOnOpHandler(
    bool update_chain, const OpInvocation& invocation,
    RCReference<Device> device, typename OpHandlerTraits::OpEntryTy op_entry,
    typename OpHandlerTraits::OpHandlerInfoTy op_handler_info) {
  return internal::ExecuteOnOpHandlerImpl<OpHandlerTraits>(
      update_chain, invocation, std::move(device), std::move(op_entry),
      op_handler_info);
}

template <typename OpHandlerTraits>
bool ExecuteOnOpHandler(bool update_chain, const OpInvocation& invocation,
                        RCReference<Device> device,
                        typename OpHandlerTraits::OpEntryTy op_entry) {
  // For now implement the non-OpHandlerInfoTy overload by faking a
  // OpHandlerInfoTy using an `int`.
  struct InnerOpHandlerTraits {
    using InputTensorTy = typename OpHandlerTraits::InputTensorTy;
    using OpEntryTy = typename OpHandlerTraits::OpEntryTy;
    using OpHandlerInfoTy = int;

    static bool MaybeConvertTensor(const OpEntryTy& op_entry, OpHandlerInfoTy,
                                   const Tensor& arg_tensor,
                                   const ExecutionContext& exec_ctx,
                                   RCReference<AsyncValue>* converted) {
      return OpHandlerTraits::MaybeConvertTensor(op_entry, arg_tensor, exec_ctx,
                                                 converted);
    }

    static void Dispatch(const OpEntryTy& op_entry, OpHandlerInfoTy,
                         ArrayRef<InputTensorTy*> inputs,
                         const OpAttrsRef& attrs,
                         ArrayRef<TensorMetadata> result_mds,
                         MutableArrayRef<RCReference<AsyncValue>> results,
                         AsyncValueRef<Chain>* chain,
                         const ExecutionContext& exec_ctx) {
      OpHandlerTraits::Dispatch(op_entry, inputs, attrs, result_mds, results,
                                chain, exec_ctx);
    }
  };

  return internal::ExecuteOnOpHandlerImpl<InnerOpHandlerTraits>(
      update_chain, invocation, std::move(device), std::move(op_entry),
      /*op_handler_info=*/0);
}

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_DISPATCH_UTILS_H_
