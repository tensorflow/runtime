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

//===- kernels.cc ---------------------------------------------------------===//
//
// This file implements kernels for distributed execution.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/payload.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_chain_manager.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_execute.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/remote_tensor.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/refcounted_callback.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {

namespace {

//===----------------------------------------------------------------------===//
// Dist AllReduce
//===----------------------------------------------------------------------===//
template <typename T>
void SumReductionFn(char* lhs, const char* rhs, size_t data_size) {
  T* typed_lhs = reinterpret_cast<T*>(lhs);
  const T* typed_rhs = reinterpret_cast<const T*>(rhs);
  const size_t num_elements = data_size / sizeof(T);
  for (size_t i = 0; i < num_elements; ++i) {
    typed_lhs[i] += typed_rhs[i];
  }
}
template <typename T>
void MaxReductionFn(char* lhs, const char* rhs, size_t data_size) {
  T* typed_lhs = reinterpret_cast<T*>(lhs);
  const T* typed_rhs = reinterpret_cast<const T*>(rhs);
  const size_t num_elements = data_size / sizeof(T);
  for (size_t i = 0; i < num_elements; ++i) {
    typed_lhs[i] = typed_lhs[i] > typed_rhs[i] ? typed_lhs[i] : typed_rhs[i];
  }
}

template <typename T>
void MinReductionFn(char* lhs, const char* rhs, size_t data_size) {
  T* typed_lhs = reinterpret_cast<T*>(lhs);
  const T* typed_rhs = reinterpret_cast<const T*>(rhs);
  const size_t num_elements = data_size / sizeof(T);
  for (size_t i = 0; i < num_elements; ++i) {
    typed_lhs[i] = typed_lhs[i] < typed_rhs[i] ? typed_lhs[i] : typed_rhs[i];
  }
}

template <typename T>
void DivFinalFn(char* lhs, size_t data_size, size_t group_size) {
  T* typed_lhs = reinterpret_cast<T*>(lhs);
  const size_t num_elements = data_size / sizeof(T);
  for (size_t i = 0; i < num_elements; ++i) {
    typed_lhs[i] = typed_lhs[i] / group_size;
  }
}

using ElementWiseReductionFunction =
    std::function<void(char* lhs, const char* rhs, size_t data_size)>;
using ElementWiseFinalFunction =
    std::function<void(char* lhs, size_t data_size, size_t group_size)>;

void IdentityFinalFn(char* lhs, size_t data_size, size_t group_size) {}

template <typename T>
llvm::StringRef GetSplit(llvm::StringRef split_input, size_t num_splits,
                         size_t num_elements, int split_id) {
  if (num_elements < num_splits) {
    TFRT_LOG(FATAL)
        << "Expected at least one element per split for num_elements="
        << num_elements << " and num_splits=" << num_splits;
  }
  llvm::ArrayRef<T> input(reinterpret_cast<const T*>(split_input.data()),
                          num_elements);
  // First compute an equitable num_elements_per_split as floor(num_elements /
  // num_splits).  Note that this can leave some elements leftover
  // (extra_elements) if num_elements does not evenly divide num_splits.  e.g.
  // if num_elements = 7 and num_splits = 4.  In this case, we spread out the
  // leftover elements equally between the first extra_elements splits.  So in
  // the previous example, extra_elements = 3, and the first 3 splits get one
  // extra element each, producing [2, 2, 2, 1].
  const size_t num_elements_per_split = num_elements / num_splits;
  const size_t extra_elements = num_elements % num_splits;
  size_t num_previous_elements;
  size_t num_elements_current_split;
  if (split_id < extra_elements) {
    // This split (and all the ones before it) get one extra element.
    num_previous_elements = split_id * (num_elements_per_split + 1);
    num_elements_current_split = num_elements_per_split + 1;
  } else {
    // This split gets num_elements_per_split items.  The ones before it got
    // (split_id * num_elements_per_split + extra_elements).
    num_previous_elements = split_id * num_elements_per_split + extra_elements;
    num_elements_current_split = num_elements_per_split;
  }
  return llvm::StringRef(
      reinterpret_cast<const char*>(input.data() + num_previous_elements),
      num_elements_current_split * sizeof(T));
}

InstanceKey StepKey(const std::string& prefix, const InstanceKey& instance_key,
                    int step) {
  return StrCat(prefix, ":", instance_key, ":", step);
}

size_t SplitIndex(int id, size_t group_size, int step) {
  size_t index = id - step;
  index = ((index % group_size) + group_size) % group_size;
  return index;
}

int FindMyIndex(
    const llvm::SmallVector<TaskHandle, 8>& collective_group_members,
    TaskHandle my_task_handle) {
  for (int i = 0; i < collective_group_members.size(); ++i) {
    if (collective_group_members[i] == my_task_handle) {
      return i;
    }
  }
  return -1;
}

TaskHandle GetTaskHandle(Argument<DistributedContext> dist_context,
                         StringAttribute task_name,
                         const ExecutionContext& exec_ctx) {
  return dist_context->GetTaskHandle(task_name.get());
}

template <typename T>
void DoAllReduce(const ExecutionContext& exec_ctx,
                 AsyncValueRef<DistributedContext> dist_ctx,
                 const InstanceKey& instance_key,
                 const std::string& collective_group_name,
                 const DenseHostTensor& in_tensor,
                 const DenseHostTensor& out_tensor,
                 ElementWiseReductionFunction reduction_fn,
                 ElementWiseFinalFunction final_fn,
                 AsyncValueRef<Chain> out_chain) {
  const auto& collective_group =
      dist_ctx->GetCollectiveGroup(collective_group_name);
  const int my_index =
      FindMyIndex(collective_group.members, dist_ctx->GetTaskHandle());
  if (my_index == -1) {
    out_chain.SetError(StrCat("The current task ", dist_ctx->GetTaskName(),
                              " is not part of the collective group ",
                              collective_group_name));
    return;
  }
  const size_t kGroupSize = collective_group.members.size();
  const size_t kLastScatterStep = kGroupSize - 1;
  const size_t kLastGatherStep = 2 * kGroupSize - 2;
  const auto kPrefix = collective_group_name;
  const int kTotalSteps = 2 * kGroupSize - 1;

  const int neighbor_index = (my_index + 1) % collective_group.members.size();
  const TaskHandle neighbor_task = collective_group.members[neighbor_index];

  auto in_tensor_ref =
      llvm::StringRef(reinterpret_cast<const char*>(in_tensor.data()),
                      in_tensor.DataSizeInBytes());
  auto* callback_registry = dist_ctx->GetCallbackRegistry();
  RemoteClientInterface* neighbor_client =
      dist_ctx->GetRemoteClient(neighbor_task);

  auto done = [out_chain = out_chain.CopyRef(),
               dist_ctx = dist_ctx.CopyRef()](Error e) mutable {
    if (e) {
      out_chain.SetError(e);
    } else {
      out_chain.emplace();
    }
  };

  // Ref counted callback to keep track of pending steps in all reduce.
  // Add one ref before starting each step, and drop one ref when the step
  // finishes (for steps with async RPCs, drop the reference when RPC finishes).
  auto refcounted_done = TakeRef(
      new RefCountedCallback([host = dist_ctx->GetHostContext(), exec_ctx,
                              done = std::move(done)](Error e) mutable {
        // NOTE: we might be executing this in either HostContext work queue
        // threads or the FabricCommunicator callback threads. Must make sure
        // AsyncValue Chain gets emplaced (or set error) in the work queue
        // threadpool, so that:
        //   * subsequent operations (i.e., AndThen) for this AsyncValue are
        //     executed in the work queue threads;
        //   * the AsyncValue drops its last ref and gets deallocated in the
        //     work queue threads
        // Otherwise, the HostContext might get destroyed before the AsyncValue
        // is deallocated or finishes its AndThen work, leading to segfault.
        if (host->IsInWorkerThread()) {
          done(std::move(e));
        } else {
          EnqueueWork(exec_ctx,
                      [done = std::move(done), e = std::move(e)]() mutable {
                        done(std::move(e));
                      });
        }
      }));

  for (int step = 0; step < kTotalSteps; ++step) {
    const InstanceKey step_key = StepKey(kPrefix, instance_key, step);
    const InstanceKey next_step_key = StepKey(kPrefix, instance_key, step + 1);
    const size_t split_id = SplitIndex(my_index, kGroupSize, step);
    llvm::StringRef split_data = GetSplit<T>(in_tensor_ref, kGroupSize,
                                             in_tensor.NumElements(), split_id);
    auto request = std::make_unique<SendDataRequest>();
    auto response = std::make_unique<SendDataResponse>();
    request->set_context_id(dist_ctx->GetContextId());
    request->set_instance_key(next_step_key);

    if (step == 0) {
      request->add_payload(split_data.data(), split_data.size());
      neighbor_client->SendDataAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request = std::move(request), response = std::move(response),
           refcounted_done = refcounted_done.CopyRef()](Error e) {
            refcounted_done->UpdateState(std::move(e));
          });
    } else if (step <= kLastScatterStep) {
      // Scatter stage: send a chunk to the neighbor, aggregate the incoming
      // chunk with local buffer.
      callback_registry->SetCallback(
          step_key,
          [step, in_split = split_data, out_split = split_data,
           request = std::move(request), response = std::move(response),
           neighbor_client, reduction_fn, final_fn, kLastScatterStep,
           kGroupSize, refcounted_done = refcounted_done.CopyRef()](
              const InstanceKey&,
              CallbackRegistry::CallbackValue callback_value) mutable {
            // Scatter aggregates the results with the local buffer.
            reduction_fn(const_cast<char*>(callback_value->data()),
                         const_cast<char*>(in_split.data()), in_split.size());

            if (step == kLastScatterStep) {
              final_fn(const_cast<char*>(callback_value->data()),
                       in_split.size(), kGroupSize);
              std::copy(callback_value->begin(), callback_value->end(),
                        const_cast<char*>(out_split.begin()));
            }
            request->add_payload(callback_value->data(),
                                 callback_value->size());
            neighbor_client->SendDataAsync(
                RemoteCallContext::GetDefault(), request.get(), response.get(),
                [request = std::move(request), response = std::move(response),
                 refcounted_done = refcounted_done.CopyRef()](Error e) mutable {
                  refcounted_done->UpdateState(std::move(e));
                });
          });
    } else {
      // Gather stage: an incoming chunk is final; just assign it to local
      // buffer and pass it to the neighbor as is.
      callback_registry->SetCallback(
          step_key,
          [step, out_split = split_data, kLastGatherStep,
           request = std::move(request), response = std::move(response),
           neighbor_client, refcounted_done = refcounted_done.CopyRef()](
              const InstanceKey&,
              CallbackRegistry::CallbackValue callback_value) mutable {
            // Gather assigns the incoming data to the local buffer
            std::copy(callback_value->begin(), callback_value->end(),
                      const_cast<char*>(out_split.begin()));
            if (step < kLastGatherStep) {
              request->add_payload(callback_value->data(),
                                   callback_value->size());
              neighbor_client->SendDataAsync(
                  RemoteCallContext::GetDefault(), request.get(),
                  response.get(),
                  [request = std::move(request), response = std::move(response),
                   refcounted_done =
                       refcounted_done.CopyRef()](Error e) mutable {
                    refcounted_done->UpdateState(std::move(e));
                  });
            }
          });
    }
  }
}

template <typename T>
void AllReduce(Argument<DistributedContext> dist_context,
               Argument<std::string> collective_group_name,
               Argument<InstanceKey> instance_key,
               Argument<DenseHostTensor> in_tensor,
               Argument<DenseHostTensor> out_tensor, Argument<Chain> in_chain,
               Result<Chain> out_chain, StringAttribute reduction_name,
               const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  ElementWiseReductionFunction reduction_fn;
  ElementWiseFinalFunction final_fn = IdentityFinalFn;
  if (reduction_name.get() == "sum") {
    reduction_fn = SumReductionFn<T>;
  } else if (reduction_name.get() == "min") {
    reduction_fn = MinReductionFn<T>;
  } else if (reduction_name.get() == "max") {
    reduction_fn = MaxReductionFn<T>;
  } else if (reduction_name.get() == "mean") {
    reduction_fn = SumReductionFn<T>;
    final_fn = DivFinalFn<T>;
  } else {
    out_chain_indirect.SetError("unexpected reduction_name in AllReduce");
    return;
  }

  EnqueueWork(exec_ctx, [exec_ctx, instance_key = *instance_key,
                         dist_context = dist_context.ValueRef(),
                         collective_group_name = *collective_group_name,
                         in_tensor = in_tensor.ValueRef(),
                         out_tensor = out_tensor.ValueRef(), reduction_fn,
                         final_fn, out_chain = std::move(out_chain_indirect)] {
    DoAllReduce<T>(exec_ctx, dist_context.CopyRef(), instance_key,
                   collective_group_name, in_tensor.get(), out_tensor.get(),
                   reduction_fn, final_fn, out_chain.CopyRef());
  });
}

template <typename T>
void DoBroadcast(AsyncValueRef<DistributedContext> dist_ctx,
                 const InstanceKey& instance_key,
                 const std::string& collective_group_name,
                 DenseHostTensor& tensor, const TaskHandle sender,
                 AsyncValueRef<Chain> out_chain) {
  const auto& collective_group =
      dist_ctx->GetCollectiveGroup(collective_group_name);
  const int my_index =
      FindMyIndex(collective_group.members, dist_ctx->GetTaskHandle());
  if (my_index == -1) {
    out_chain.SetError(StrCat("The current task ", dist_ctx->GetTaskName(),
                              " is not part of the collective group ",
                              collective_group_name));
    return;
  }
  const auto kGroupSize = collective_group.members.size();
  const auto kPrefix = collective_group_name;
  const TaskHandle my_task = dist_ctx->GetTaskHandle();
  const int neighbor_index = (my_index + 1) % kGroupSize;
  const TaskHandle neighbor_task = collective_group.members[neighbor_index];

  auto in_tensor = llvm::StringRef(reinterpret_cast<const char*>(tensor.data()),
                                   tensor.DataSizeInBytes());
  const auto num_elements = tensor.NumElements();

  auto* registry = dist_ctx->GetCallbackRegistry();
  RemoteClientInterface* neighbor_client =
      dist_ctx->GetRemoteClient(neighbor_task);

  auto refcounted_done = TakeRef(
      new RefCountedCallback([out_chain = out_chain.CopyRef(),
                              dist_ctx = dist_ctx.CopyRef()](Error e) mutable {
        if (e) {
          out_chain.SetError(e);
        } else {
          out_chain.emplace();
        }
      }));

  for (auto i = 0; i < kGroupSize; ++i) {
    auto chunk_key = StepKey(kPrefix, instance_key, i);
    auto request = std::make_unique<SendDataRequest>();
    auto response = std::make_unique<SendDataResponse>();
    request->set_context_id(dist_ctx->GetContextId());
    request->set_instance_key(StepKey(kPrefix, chunk_key, neighbor_index));
    if (my_task == sender) {
      // A Sender sends data to its neighbor.
      auto payload = GetSplit<T>(in_tensor, kGroupSize, num_elements, i);
      request->add_payload(payload.data(), payload.size());
      neighbor_client->SendDataAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request = std::move(request), response = std::move(response),
           refcounted_done = refcounted_done.CopyRef()](Error e) {
            refcounted_done->UpdateState(std::move(e));
          });
    } else {
      registry->SetCallback(
          StepKey(kPrefix, chunk_key, my_index),
          [sender, i, in_tensor, kGroupSize, neighbor_task, num_elements,
           neighbor_client, request = std::move(request),
           response = std::move(response),
           refcounted_done = refcounted_done.CopyRef()](
              const InstanceKey&,
              CallbackRegistry::CallbackValue data) mutable {
            // A neighbor receives data and forwards it to its neighbor.
            std::copy(data->begin(), data->end(),
                      const_cast<char*>(
                          GetSplit<T>(in_tensor, kGroupSize, num_elements, i)
                              .begin()));
            if (neighbor_task != sender) {
              request->add_payload(data->data(), data->size());
              neighbor_client->SendDataAsync(
                  RemoteCallContext::GetDefault(), request.get(),
                  response.get(),
                  [request = std::move(request), response = std::move(response),
                   refcounted_done = refcounted_done.CopyRef()](Error e) {
                    refcounted_done->UpdateState(std::move(e));
                  });
            }
          });
    }
  }
}

template <typename T>
void Broadcast(Argument<DistributedContext> dist_context,
               Argument<std::string> collective_group_name,
               Argument<InstanceKey> instance_key,
               Argument<DenseHostTensor> in_tensor, Argument<TaskHandle> sender,
               Argument<Chain> in_chain, Result<Chain> out_chain,
               const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  EnqueueWork(exec_ctx, [dist_context = dist_context.ValueRef(),
                         instance_key = *instance_key,
                         collective_group_name = *collective_group_name,
                         sender = *sender, in_tensor_ref = in_tensor.ValueRef(),
                         out_chain = std::move(out_chain_indirect)] {
    DoBroadcast<T>(dist_context.CopyRef(), instance_key, collective_group_name,
                   in_tensor_ref.get(), sender, out_chain.CopyRef());
  });
}

// AllGather functionality for both AllGatherFixed and AllGatherAny.
// For AllGatherFixed, sizes will be empty.
// For AllGatherAny, sizes will be non-empty.
template <typename T>
void DoAllGather(const ExecutionContext& exec_ctx,
                 AsyncValueRef<DistributedContext> dist_ctx,
                 const InstanceKey& instance_key,
                 const CollectiveGroup& collective_group, const int my_index,
                 const DenseHostTensor& in_tensor, DenseHostTensor& out_tensor,
                 RCReference<RefCountedCallback> refcounted_done,
                 MutableArrayRef<size_t> sizes = {}) {
  const auto kGroupSize = collective_group.members.size();
  const auto kNeighborIndex = (my_index + 1) % kGroupSize;
  const auto kNeighborId = collective_group.members[kNeighborIndex];
  const auto kPrefix = collective_group.name;
  const auto kContextId = dist_ctx->GetContextId();
  auto in_tensor_ref =
      llvm::StringRef(reinterpret_cast<const char*>(in_tensor.data()),
                      in_tensor.DataSizeInBytes());
  auto out_tensor_ref = reinterpret_cast<char*>(out_tensor.data());
  const auto num_elements = in_tensor.NumElements();
  auto* registry = dist_ctx->GetCallbackRegistry();
  RemoteClientInterface* neighbor_client =
      dist_ctx->GetRemoteClient(kNeighborId);

  for (size_t ring_order = 0; ring_order < kGroupSize; ++ring_order) {
    const auto chunk_key = StepKey(kPrefix, instance_key, ring_order);
    const size_t pos =
        sizes.empty() ? ring_order * num_elements : sizes[ring_order];
    auto request = std::make_unique<SendDataRequest>();
    auto response = std::make_unique<SendDataResponse>();
    request->set_context_id(kContextId);
    request->set_instance_key(StepKey(kPrefix, chunk_key, kNeighborIndex));
    if (my_index == ring_order) {
      std::copy(in_tensor_ref.begin(), in_tensor_ref.end(),
                out_tensor_ref + pos * sizeof(T));
      request->add_payload(in_tensor_ref.data(), in_tensor_ref.size());
      neighbor_client->SendDataAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request = std::move(request), response = std::move(response),
           refcounted_done = refcounted_done.CopyRef()](Error e) {
            refcounted_done->UpdateState(std::move(e));
          });

    } else {
      registry->SetCallback(
          StepKey(kPrefix, chunk_key, my_index),
          [ring_order, out_tensor_ref, pos, kNeighborIndex, neighbor_client,
           request = std::move(request), response = std::move(response),
           refcounted_done = refcounted_done.CopyRef()](
              const InstanceKey&,
              CallbackRegistry::CallbackValue callback_value) mutable {
            // A neighbor receives data and forwards it to its neighbor.
            std::copy(callback_value->begin(), callback_value->end(),
                      out_tensor_ref + pos * sizeof(T));
            if (ring_order != kNeighborIndex) {
              request->add_payload(callback_value->data(),
                                   callback_value->size());
              neighbor_client->SendDataAsync(
                  RemoteCallContext::GetDefault(), request.get(),
                  response.get(),
                  [request = std::move(request), response = std::move(response),
                   refcounted_done = refcounted_done.CopyRef()](Error e) {
                    refcounted_done->UpdateState(std::move(e));
                  });
            }
          });
    }
  }
}

template <typename T>
void AllGatherFixedShape(Argument<DistributedContext> dist_ctx,
                         Argument<std::string> collective_group_name,
                         Argument<InstanceKey> instance_key,
                         Argument<DenseHostTensor> in_tensor,
                         Argument<DenseHostTensor> out_tensor,
                         Argument<Chain> in_chain, Result<Chain> out_chain,
                         const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  const auto& collective_group =
      dist_ctx->GetCollectiveGroup(collective_group_name.get());
  int my_index =
      FindMyIndex(collective_group.members, dist_ctx->GetTaskHandle());
  if (my_index == -1) {
    out_chain_indirect.SetError(
        "This worker is not part of the collective group ");
    return;
  }
  auto* registry = dist_ctx->GetCallbackRegistry();
  auto done = [out_tensor = out_tensor.ValueRef(), registry,
               instance_key = *instance_key,
               out_chain = std::move(out_chain_indirect),
               dist_ctx = dist_ctx.ValueRef()](Error e) mutable {
    if (e) {
      out_chain.SetError(e);
    } else {
      out_chain.emplace();
    }
    registry->SetValue(instance_key, /*value=*/nullptr);
  };
  auto refcounted_done = TakeRef(
      new RefCountedCallback([host = dist_ctx->GetHostContext(), exec_ctx,
                              done = std::move(done)](Error e) mutable {
        if (host->IsInWorkerThread()) {
          done(std::move(e));
        } else {
          EnqueueWork(exec_ctx,
                      [done = std::move(done), e = std::move(e)]() mutable {
                        done(std::move(e));
                      });
        }
      }));

  EnqueueWork(exec_ctx, [exec_ctx, my_index, instance_key = *instance_key,
                         collective_group, dist_ctx = dist_ctx.ValueRef(),
                         in_tensor_ref = in_tensor.ValueRef(),
                         out_tensor_ref = out_tensor.ValueRef(),
                         refcounted_done = refcounted_done.CopyRef()] {
    DoAllGather<T>(exec_ctx, dist_ctx.CopyRef(), instance_key, collective_group,
                   my_index, in_tensor_ref.get(), out_tensor_ref.get(),
                   refcounted_done.CopyRef());
  });
}

void RemoteRegisterKernelHelper(Chain ch, DistributedContext* dist_context,
                                const TaskHandle receiver,
                                RemainingResults results,
                                StringAttribute program,
                                StringAttribute program_name,
                                bool need_compilation,
                                const ExecutionContext& exec_ctx) {
  auto request = std::make_unique<RegisterFunctionRequest>();
  // program and program_name will live as long as out_chain is not populated.
  request->set_context_id(dist_context->GetContextId());
  request->set_program(program.str());
  request->set_program_name(program_name.str());
  request->set_need_compilation(need_compilation);
  RemoteClientInterface* remote_client =
      dist_context->GetRemoteClient(receiver);

  RCReference<AsyncValue> out;
  if (need_compilation) {
    out = MakeUnconstructedAsyncValueRef<RemoteExecuteSpec>(exec_ctx.host());
  } else {
    out = MakeUnconstructedAsyncValueRef<Chain>(exec_ctx.host());
  }
  results[0] = out.CopyRef();

  EnqueueWork(exec_ctx, [remote_client, request = std::move(request),
                         dist_context, need_compilation,
                         out = out.CopyRef()]() mutable {
    auto response = std::make_unique<RegisterFunctionResponse>();
    remote_client->RegisterFunctionAsync(
        RemoteCallContext::GetDefault(), request.get(), response.get(),
        [request = std::move(request), response = std::move(response),
         need_compilation, dist_context, out = out.CopyRef()](Error e) mutable {
          if (e) {
            out->SetError(DecodedDiagnostic(std::move(e)));
          } else {
            if (need_compilation) {
              DeviceManager* manager = dist_context->GetRemoteDeviceManager();
              llvm::SmallVector<RCReference<Device>, 4> output_devices;
              output_devices.reserve(response->output_device_size());
              for (int i = 0; i < response->output_device_size(); i++) {
                RCReference<Device> device =
                    manager->GetDeviceRef<Device>(response->output_device(i));
                output_devices.push_back(device.CopyRef());
              }
              out->emplace<RemoteExecuteSpec>(std::move(output_devices));
            } else {
              out->emplace<Chain>();
            }
          }
        });
  });
}

void RegisterTFRTFunctionKernel(Chain ch, DistributedContext* dist_context,
                                TaskHandle receiver, RemainingResults results,
                                StringAttribute program,
                                StringAttribute program_name,
                                const ExecutionContext& exec_ctx) {
  RemoteRegisterKernelHelper(ch, dist_context, receiver, results, program,
                             program_name, /*need_compilation=*/false,
                             exec_ctx);
}

void RegisterTFFunctionKernel(Chain ch, DistributedContext* dist_context,
                              TaskHandle receiver, RemainingResults results,
                              StringAttribute program,
                              StringAttribute program_name,
                              const ExecutionContext& exec_ctx) {
  RemoteRegisterKernelHelper(ch, dist_context, receiver, results, program,
                             program_name, /*need_compilation=*/true, exec_ctx);
}

AsyncValueRef<RemoteExecuteSpec> CreateRemoteExecuteSpec(
    DistributedContext* dist_context, AggregateAttr inputs,
    const ExecutionContext& exec_ctx) {
  llvm::SmallVector<RCReference<Device>, 4> output_devices;
  output_devices.reserve(inputs.GetNumElements());
  for (int i = 0, e = inputs.GetNumElements(); i < e; ++i) {
    const std::string& device_str =
        inputs.GetAttributeOfType<StringAttr>(i).GetValue().str();
    RCReference<Device> device =
        dist_context->GetRemoteDeviceManager()->GetDeviceRef<Device>(
            device_str);
    if (device.get() == nullptr) {
      TFRT_LOG(ERROR) << "Can't find device: " << device_str;
      return MakeErrorAsyncValueRef(exec_ctx.host(),
                                    StrCat("Can't find device: ", device_str));
    }
    output_devices.push_back(device.CopyRef());
  }
  return MakeAvailableAsyncValueRef<RemoteExecuteSpec>(
      exec_ctx.host(), std::move(output_devices));
}

void RemoteExecute(Chain ch, DistributedContext* dist_context,
                   const TaskHandle receiver, Argument<RemoteExecuteSpec> spec,
                   RemainingArguments inputs, RemainingResults results,
                   StringAttribute program_name, int num_fn_inputs,
                   int32_t num_output_with_tensorhandle,
                   const ExecutionContext& exec_ctx) {
  // If some output IDs are present in the inputs, we assume all output IDs are
  // pre-allocated.
  const bool output_id_allocated = num_fn_inputs != inputs.size();
  auto request = std::make_unique<RemoteExecuteRequest>();
  // program_name will live as long as out_chain is not populated.
  request->set_context_id(dist_context->GetContextId());
  request->set_program_name(program_name.str());
  request->mutable_input()->Reserve(num_fn_inputs);

  for (int i = 0; i < num_fn_inputs; ++i) {
    const RemoteObjectId& input = inputs[i]->get<RemoteObjectId>();
    auto* add_input = request->add_input();
    add_input->set_prefix_id(input.prefix_id);
    add_input->set_local_id(input.local_id);
    add_input->set_device(input.device->name().str());
  }
  // First output: chain
  AsyncValueRef<Chain> out_chain =
      MakeConstructedAsyncValueRef<Chain>(exec_ctx.host());
  results[0] = out_chain.CopyRef();

  // If output_id is preallocated, we only return TensorHandles. Otherwise, we
  // return output ids followed by TensorHandles.
  if (results.size() !=
      1 /*chain*/ + (output_id_allocated ? 0 : spec->output_devices.size()) +
          num_output_with_tensorhandle) {
    out_chain.SetError(llvm::make_error<InvalidArgumentErrorInfo>(StrCat(
        "Mismatch output devices size in RemoteExecuteSpec: ",
        spec->output_devices.size(), " expected: ", results.size() - 1)));
    return;
  }

  // Actual number of outputs of the remote function.
  int num_fn_output;
  if (output_id_allocated) {
    // Each of the output IDs must be passed as inputs.
    num_fn_output = inputs.size() - num_fn_inputs;
  } else {
    // Otherwise, we can infer this from the kernel outputs minus chain minus
    // TensorHandle output
    num_fn_output = results.size() - num_output_with_tensorhandle - 1;
  }
  // Start output index of TensorHandle outputs.
  // If output id is allocated, we only return TensorHandles.
  const int th_output_idx = output_id_allocated ? 0 : num_fn_output;
  request->mutable_output()->Reserve(num_fn_output);
  RemoteObjectManager* manager = dist_context->GetRemoteObjectManager();
  struct RemoteObjectAndMetadata {
    AsyncValueRef<RemoteObjectId> id;
    AsyncValueRef<RemoteTensor> tensor;
    AsyncValueRef<TensorMetadata> metadata;
  };
  llvm::SmallVector<RemoteObjectAndMetadata, 4> remote_objs;
  for (int i = 1; i <= num_fn_output; ++i) {
    RCReference<Device> output_device = spec->output_devices[i - 1].CopyRef();
    AsyncValueRef<RemoteObjectId> out_id;
    if (output_id_allocated) {
      // Reuse output id
      out_id =
          AsyncValueRef<RemoteObjectId>(FormRef(inputs[num_fn_inputs + i - 1]));
    } else {
      // Allocate output id
      out_id = MakeAvailableAsyncValueRef<RemoteObjectId>(
          exec_ctx.host(),
          manager->AllocateRemoteObject(std::move(output_device)));
      // The next num_id_outputs are RemoteObjectId
      results[i] = out_id.CopyRef();
    }
    // The last num_output_with_metadata RemoteObjectIds needs to have
    // TensorMetadata returned.
    const bool need_metadata =
        i > (num_fn_output - num_output_with_tensorhandle);
    if (need_metadata) {
      auto tensor =
          MakeUnconstructedAsyncValueRef<RemoteTensor>(exec_ctx.host());
      auto metadata =
          MakeUnconstructedAsyncValueRef<TensorMetadata>(exec_ctx.host());
      AsyncValueRef<TensorHandle> th = MakeAvailableAsyncValueRef<TensorHandle>(
          exec_ctx.host(), out_id->device.CopyRef(), metadata.CopyRef(),
          tensor.CopyRef());
      remote_objs.emplace_back(RemoteObjectAndMetadata{
          out_id.CopyRef(), std::move(tensor), std::move(metadata)});
      // The remaining outputs are TensorHandle
      results[th_output_idx + remote_objs.size()] = th.CopyRef();
    }
    auto* add_output = request->add_output();
    add_output->set_need_metadata(need_metadata);
    auto* add_output_id = add_output->mutable_id();
    add_output_id->set_prefix_id(out_id->prefix_id);
    add_output_id->set_local_id(out_id->local_id);
    add_output_id->set_device(out_id->device->name().str());
  }

  RemoteClientInterface* remote_client =
      dist_context->GetRemoteClient(receiver);
  EnqueueWork(exec_ctx, [remote_client, request = std::move(request),
                         dist_context, out_chain = out_chain.CopyRef(),
                         remote_objs = std::move(remote_objs)]() mutable {
    auto response = std::make_unique<RemoteExecuteResponse>();
    remote_client->RemoteExecuteAsync(
        RemoteCallContext::GetDefault(), request.get(), response.get(),
        [request = std::move(request), response = std::move(response),
         out_chain = out_chain.CopyRef(), remote_objs = std::move(remote_objs),
         host_context = dist_context->GetHostContext()](Error e) mutable {
          // Propagate metadata and output chain
          const int num_metadata = response->metadata_size();
          for (int i = 0; i < remote_objs.size(); ++i) {
            auto& obj = remote_objs[i];
            if (i >= num_metadata) {
              obj.metadata.SetError(DecodedDiagnostic("Metadata not returned"));
              continue;
            }
            auto metadata = DeserializeTensorMetadata(response->metadata(i));
            if (metadata) {
              obj.metadata.emplace(metadata.get());
              obj.tensor.emplace(std::move(metadata.get()), obj.id.get());
            } else {
              obj.tensor.SetError(DecodedDiagnostic(metadata.takeError()));
            }
          }
          if (e) {
            out_chain.SetError(std::move(e));
          } else {
            out_chain.SetStateConcrete();
          }
        });
  });
}

void RemoteExecuteKernel(Chain ch, DistributedContext* dist_context,
                         TaskHandle receiver, Argument<RemoteExecuteSpec> spec,
                         RemainingArguments inputs, RemainingResults results,
                         StringAttribute program_name,
                         const ExecutionContext& exec_ctx) {
  RemoteExecute(ch, dist_context, receiver, spec, inputs, results, program_name,
                inputs.size(), 0, exec_ctx);
}

void RemoteExecuteTHKernel(Chain ch, DistributedContext* dist_context,
                           TaskHandle receiver,
                           Argument<RemoteExecuteSpec> spec,
                           RemainingArguments inputs, RemainingResults results,
                           Attribute<int32_t> num_output_with_tensorhandle,
                           StringAttribute program_name,
                           const ExecutionContext& exec_ctx) {
  RemoteExecute(ch, dist_context, receiver, spec, inputs, results, program_name,
                inputs.size(), num_output_with_tensorhandle.get(), exec_ctx);
}

void RemoteExecuteTHPreallocatedKernel(
    Chain ch, DistributedContext* dist_context, TaskHandle receiver,
    Argument<RemoteExecuteSpec> spec, RemainingArguments inputs,
    RemainingResults results, Attribute<int32_t> num_inputs,
    Attribute<int32_t> num_output_with_tensorhandle,
    StringAttribute program_name, const ExecutionContext& exec_ctx) {
  RemoteExecute(ch, dist_context, receiver, spec, inputs, results, program_name,
                num_inputs.get(), num_output_with_tensorhandle.get(), exec_ctx);
}

AsyncValueRef<RemoteObjectId> GetChainForTaskHandle(
    Chain ch, RemoteChainManager* chain_manager, TaskHandle task,
    const ExecutionContext& exec_ctx) {
  return MakeAvailableAsyncValueRef<RemoteObjectId>(
      exec_ctx.host(), chain_manager->GetRemoteChain(task));
}

AsyncValueRef<Chain> SetChainForTaskHandle(
    Chain ch, RemoteChainManager* chain_manager, TaskHandle task,
    Argument<RemoteObjectId> remote_chain, const ExecutionContext& exec_ctx) {
  chain_manager->SetRemoteChain(task, remote_chain.get());
  return MakeAvailableAsyncValueRef<Chain>(exec_ctx.host());
}

AsyncValueRef<DistributedContext> GetDistributedContext(
    const ExecutionContext& exec_ctx) {
  return exec_ctx.request_ctx()
      ->GetData<AsyncValueRef<DistributedContext>>()
      .CopyRef();
}

void SendBytes(Argument<DistributedContext> dist_context,
               Argument<TaskHandle> receiver_task,
               Argument<InstanceKey> instance_key, Argument<Payload> serialized,
               Result<Chain> out_chain, const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  auto request = std::make_unique<SendDataRequest>();
  auto response = std::make_unique<SendDataResponse>();
  request->set_context_id(dist_context->GetContextId());
  request->set_instance_key(*instance_key);
  const int kNumBuffers = serialized->buffers.size();
  // TODO(pisong): avoid copying here
  for (size_t i = 0; i < kNumBuffers; ++i) {
    request->add_payload(serialized->buffers[i]->data(),
                         serialized->buffers[i]->size());
  }
  dist_context->GetRemoteClient(*receiver_task)
      ->SendDataAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request = std::move(request), response = std::move(response),
           dist_context = dist_context.ValueRef(),
           out_chain = out_chain_indirect.CopyRef()](Error e) {
            if (e) {
              out_chain.SetError(e);
            } else {
              out_chain.emplace();
            }
          });
}

void RecvBytes(Argument<DistributedContext> dist_context,
               Argument<TaskHandle> sender_task,
               Argument<InstanceKey> instance_key,
               Result<std::string> serialized,
               const ExecutionContext& exec_ctx) {
  auto on_done = [serialized = serialized.Allocate()](
                     const InstanceKey&,
                     CallbackRegistry::CallbackValue value) {
    serialized.emplace(*value);
  };
  CallbackRegistry* registry = dist_context->GetCallbackRegistry();
  registry->SetCallback(*instance_key, std::move(on_done));
}

void Serialize(Argument<DenseHostTensor> dht, Argument<Chain> chain,
               Result<Payload> serialized_dht,
               const ExecutionContext& exec_ctx) {
  auto buffers = SerializeDenseHostTensor(dht.get(), exec_ctx.host());
  serialized_dht.Emplace(Payload(std::move(buffers.get())));
}

void Deserialize(Argument<std::string> serialized_dht,
                 Result<DenseHostTensor> dht,
                 const ExecutionContext& exec_ctx) {
  auto t = DeserializeDenseHostTensor(serialized_dht.get(), exec_ctx.host());
  dht.Emplace(std::move(*t));
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
void RegisterDistributedKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_dist.get_task_handle", TFRT_KERNEL(GetTaskHandle));
  registry->AddKernel("tfrt_dist.cpu.allreduce.f32",
                      TFRT_KERNEL(AllReduce<float>));
  registry->AddKernel("tfrt_dist.cpu.allreduce.i32",
                      TFRT_KERNEL(AllReduce<int32_t>));
  registry->AddKernel("tfrt_dist.cpu.broadcast.f32",
                      TFRT_KERNEL(Broadcast<float>));
  registry->AddKernel("tfrt_dist.cpu.broadcast.i32",
                      TFRT_KERNEL(Broadcast<int32_t>));
  registry->AddKernel("tfrt_dist.cpu.allgather_fixed_shape.f32",
                      TFRT_KERNEL(AllGatherFixedShape<float>));
  registry->AddKernel("tfrt_dist.cpu.allgather_fixed_shape.i32",
                      TFRT_KERNEL(AllGatherFixedShape<int32_t>));
  registry->AddKernel("tfrt_dist.create_remote_execute_spec",
                      TFRT_KERNEL(CreateRemoteExecuteSpec));
  registry->AddKernel("tfrt_dist.remote_execute",
                      TFRT_KERNEL(RemoteExecuteKernel));
  registry->AddKernel("tfrt_dist.remote_execute_th",
                      TFRT_KERNEL(RemoteExecuteTHKernel));
  registry->AddKernel("tfrt_dist.remote_execute_th_preallocated",
                      TFRT_KERNEL(RemoteExecuteTHPreallocatedKernel));
  registry->AddKernel("tfrt_dist.register_tfrt_function",
                      TFRT_KERNEL(RegisterTFRTFunctionKernel));
  registry->AddKernel("tfrt_dist.register_tf_function",
                      TFRT_KERNEL(RegisterTFFunctionKernel));
  registry->AddKernel("tfrt_dist.get_chain_for_task_handle",
                      TFRT_KERNEL(GetChainForTaskHandle));
  registry->AddKernel("tfrt_dist.set_chain_for_task_handle",
                      TFRT_KERNEL(SetChainForTaskHandle));
  registry->AddKernel("tfrt_dist.get_distributed_context",
                      TFRT_KERNEL(GetDistributedContext));
  registry->AddKernel("tfrt_dist.send_bytes", TFRT_KERNEL(SendBytes));
  registry->AddKernel("tfrt_dist.recv_bytes", TFRT_KERNEL(RecvBytes));
  registry->AddKernel("tfrt_dist.serialize", TFRT_KERNEL(Serialize));
  registry->AddKernel("tfrt_dist.deserialize", TFRT_KERNEL(Deserialize));
}

}  // namespace tfrt
