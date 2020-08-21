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

//===- dispatch_utils.cc ----------------------------------------*- C++ -*-===//
//
// This file contains some common op handler agnostic utilities for executing
// metadata and dispatch functions.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/dispatch_utils.h"

#include "tfrt/host_context/host_context.h"

namespace tfrt {
namespace internal {
MDFunctionExecResult ExecuteMetadataFunction(
    const OpMetadataFn& metadata_fn, const OpInvocation& invocation,
    SmallVectorImpl<TensorMetadata>& result_mds) {
  auto propagate_error = [&](RCReference<AsyncValue> error) {
    for (auto& result : invocation.results)
      result = TensorHandle::CreateError(error.CopyRef());
    if (invocation.chain)
      *invocation.chain = AsyncValueRef<Chain>(std::move(error));
  };

  // In the vastly most common case, all of the argument TensorHandles have
  // available metadata records. If so, we can synchronously execute the
  // metadata function for this op. If not, we have to fall back to running
  // the metadata function asynchronously - and enqueue the execution of the
  // kernel implementation to run when the shape results are resolved and
  // chain is available.
  SmallVector<TensorMetadata, 4> argument_mds;
  auto arguments = invocation.arguments;
  argument_mds.reserve(arguments.size());
  for (size_t i = 0, e = arguments.size(); i != e; ++i) {
    // Fetch the metadata if it is stored inline.
    if (arguments[i].IsMetadataAvailable()) {
      argument_mds.push_back(arguments[i].GetAvailableMetadata());
      continue;
    }
    const auto& arg_md_av = arguments[i].GetAsyncMetadata();

    auto arg_state = arg_md_av.GetAsyncValue()->state();
    if (arg_state.IsUnconstructed()) {
      return MDFunctionExecResult::kMetadataUnavailable;
    }

    // If any input is an error, then propagate the error and bail out early.
    if (arg_state.IsError()) {
      propagate_error(arg_md_av.CopyRef());
      return MDFunctionExecResult::kError;
    }

    // Otherwise, we have the metadata for the input.
    assert(arg_state.IsConcrete());
    argument_mds.push_back(arg_md_av.get());
  }

  // Okay, the shapes are available as we expect, get the result metadata.
  result_mds.resize(invocation.results.size());

  // TODO(tfrt-devs): Remove this tracing tag when finished debugging
  // dispatch performance.
  TFRT_TRACE_SCOPE("RunMetadataFunction");
  if (auto error = metadata_fn(invocation.exec_ctx, argument_mds,
                               invocation.attrs, result_mds)) {
    // If the metadata function produced an error, propagate it.
    propagate_error(std::move(error));
    return MDFunctionExecResult::kError;
  }

  return MDFunctionExecResult::kSuccess;
}

void ExecuteWhenMetadataIsReady(const OpInvocation& invocation,
                                const OpMetadataFn& metadata_fn,
                                bool update_chain,
                                RCReference<Device> retval_device,
                                MetadataIsReadyCallback callback) {
  auto arguments = invocation.arguments;
  auto results = invocation.results;

  SmallVector<AsyncValue*, 8> async_mds;
  async_mds.reserve(arguments.size());
  SmallVector<TensorHandle, 4> arguments_copy;
  arguments_copy.reserve(arguments.size());
  for (size_t i = 0, e = arguments.size(); i != e; ++i) {
    // Collect the unavailable async metadata values that caused us to get into
    // this slow path.  We hand these off to RunWhenReady.
    if (!arguments[i].IsMetadataAvailable()) {
      // If metadata is not available, metadata must be async, and not inline.
      const AsyncValueRef<TensorMetadata>& md = arguments[i].GetAsyncMetadata();
      async_mds.push_back(md.GetAsyncValue());
    }

    // We need to take the arguments so they are guaranteed to live for the
    // duration of the RunWhenReady closure.
    arguments_copy.push_back(std::move(arguments[i]));
  }

  // Our lambda will produce two AsyncValue's for each TensorHandle result - one
  // is the metadata result, and one is the tensor result.
  SmallVector<RCReference<AsyncValue>, 8> result_th_avs;
  result_th_avs.reserve(results.size() * 2);

  auto host = invocation.exec_ctx.host();

  // Prepopulate result_th_avs's.
  for (auto& result : results) {
    // We know the metadata value will be a TensorMetadata or an error.
    auto md = MakeUnconstructedAsyncValueRef<TensorMetadata>(host);

    // We don't know what subclass of Tensor will be used, so we need to use an
    // IndirectAsyncValue.
    auto tensor = MakeIndirectAsyncValue(host);
    result_th_avs.push_back(md.CopyRCRef());
    result_th_avs.push_back(tensor.CopyRef());
    result = TensorHandle(retval_device.CopyRef(), std::move(md),
                          AsyncValueRef<Tensor>(std::move(tensor)));
  }

  // If the op implementation has side effects, we must fulfill the chain result
  // upon completion, so we need to allocate an unavailable AsyncValue for this.
  // On the other hand, most ops don't have side effects, and we don't want to
  // serialize ops without side effects, so only allocate the chain when needed.
  AsyncValueRef<Chain> chain_ref;

  if (update_chain) {
    if (!invocation.chain->IsAvailable())
      async_mds.push_back(invocation.chain->GetAsyncValue());

    chain_ref = MakeUnconstructedAsyncValueRef<Chain>(host);
    *invocation.chain = chain_ref.CopyRef();
  }

  host->RunWhenReady(
      async_mds,
      [metadata_fn, callback = std::move(callback),
       exec_ctx = invocation.exec_ctx, frozen_attrs = invocation.attrs.freeze(),
       chain = std::move(chain_ref), result_th_avs = std::move(result_th_avs),
       arguments = std::move(arguments_copy)]() mutable {
        auto num_results = result_th_avs.size() / 2;

        // If any error is detected, this closure ties off our state and
        // propagates the error correctly.
        auto propagate_error = [&](AsyncValue* error_av) {
          auto& diag = error_av->GetError();
          // Set the previously allocated metadata AV to the error.
          for (auto& result_th_av : result_th_avs) result_th_av->SetError(diag);
          if (chain) chain.SetError(diag);
        };

        // This lambda will run when all of the async_shapes are resolved,
        // allowing us to run the shape function and then carry on.
        SmallVector<TensorMetadata, 4> argument_mds;
        argument_mds.reserve(arguments.size());
        for (size_t i = 0, e = arguments.size(); i != e; ++i) {
          // If any input is an error, then propagate the error to all outputs
          // and we are done.
          if (arguments[i].IsMetadataError()) {
            return propagate_error(
                arguments[i].GetAsyncMetadata().GetAsyncValue());
          }

          // Otherwise, we have the metadata for the input.
          argument_mds.push_back(arguments[i].GetAvailableMetadata());
        }

        // Okay, the shapes are available as we expect, run the metadata
        // function to get the result shapes.
        SmallVector<TensorMetadata, 4> result_mds(num_results);
        if (auto error =
                metadata_fn(exec_ctx, argument_mds, frozen_attrs, result_mds)) {
          // If the metadata function produced an error, propagate it.
          return propagate_error(error.get());
        }

        // Now that we know the metadata results for this op, we can fulfill the
        // AsyncValue's for the result TensorHandles.  Do this eagerly to keep
        // the shape computations flowing fast.
        for (size_t i = 0; i != num_results; ++i)
          result_th_avs[i * 2]->emplace<TensorMetadata>(result_mds[i]);

        // Now that we have the result shapes, we can run/enqueue the kernel.
        SmallVector<AsyncValueRef<Tensor>, 8> result_tensor_avs;
        callback(exec_ctx, arguments, frozen_attrs, result_mds.size(),
                 result_mds, &result_tensor_avs, &chain);

        // Now that we have the AsyncValue's for the result tensors, we can fill
        // in the IndirectAsyncValue's for the TensorHandle results.
        for (size_t i = 0; i != num_results; ++i) {
          auto* indirect_av =
              cast<IndirectAsyncValue>(result_th_avs[i * 2 + 1].get());
          indirect_av->ForwardTo(std::move(result_tensor_avs[i]));
        }
      });
}

}  // namespace internal
}  // namespace tfrt
