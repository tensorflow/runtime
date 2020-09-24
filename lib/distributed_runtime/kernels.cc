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
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/distributed_context.h"
#include "tfrt/distributed_runtime/distributed_kernels.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/remote_execute.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/logging.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

namespace {

// TODO(b/168132685): Use Status input for async callback function.
using CallbackFn = llvm::unique_function<void(bool /* success */)>;

class RefCountedCallback : public ReferenceCounted<RefCountedCallback> {
 public:
  explicit RefCountedCallback(CallbackFn done) : done_(std::move(done)) {}
  ~RefCountedCallback() { done_(is_ok_); }
  void UpdateState(const bool s) {
    mutex_lock l(mu_);
    is_ok_ = is_ok_ && s;
  }

 private:
  CallbackFn done_;
  mutex mu_;
  bool is_ok_ TFRT_GUARDED_BY(mu_) = true;
};

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

size_t SplitIndex(HostId id, size_t group_size, int step) {
  size_t index = id - step;
  index = ((index % group_size) + group_size) % group_size;
  return index;
}

int FindMyIndex(CollectiveGroup& collective_group, HostId my_id) {
  for (int i = 0; i < collective_group.members.size(); ++i) {
    if (collective_group.members[i] == my_id) {
      return i;
    }
  }
  return -1;
}

CollectiveGroup CreateCollectiveGroup(Argument<DistributedContext> dist_context,
                                      Argument<std::string> name,
                                      const ExecutionContext& exec_ctx) {
  auto collective_group = dist_context->GetCollectiveGroup(name.get());
  return collective_group;
}

template <typename T>
void DoAllReduce(const ExecutionContext& exec_ctx,
                 AsyncValueRef<DistributedContext> dist_ctx,
                 const InstanceKey& instance_key,
                 const CollectiveGroup& collective_group,
                 const DenseHostTensor& in_tensor,
                 const DenseHostTensor& out_tensor,
                 ElementWiseReductionFunction reduction_fn,
                 ElementWiseFinalFunction final_fn, HostId my_id,
                 HostId neighbor_id, AsyncValueRef<Chain> out_chain) {
  const size_t kGroupSize = collective_group.members.size();
  const size_t kLastScatterStep = kGroupSize - 1;
  const size_t kLastGatherStep = 2 * kGroupSize - 2;
  const auto kPrefix = collective_group.name;
  const int kTotalSteps = 2 * kGroupSize - 1;

  auto in_tensor_ref =
      llvm::StringRef(reinterpret_cast<const char*>(in_tensor.data()),
                      in_tensor.DataSizeInBytes());
  auto* callback_registry = dist_ctx->GetCallbackRegistry();
  auto* fabric_communicator = dist_ctx->GetOrCreateFabricCommunicator();

  auto done = [out_chain = out_chain.CopyRef(),
               dist_ctx = dist_ctx.CopyRef()](const bool ok) mutable {
    if (!ok) {
      out_chain.SetError("Failed executing all-reduce.");
    }
    out_chain.emplace();
  };

  // Ref counted callback to keep track of pending steps in all reduce.
  // Add one ref before starting each step, and drop one ref when the step
  // finishes (for steps with async RPCs, drop the reference when RPC finishes).
  auto refcounted_done = TakeRef(
      new RefCountedCallback([host = dist_ctx->GetHostContext(), exec_ctx,
                              done = std::move(done)](const bool ok) mutable {
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
          done(ok);
        } else {
          EnqueueWork(exec_ctx,
                      [done = std::move(done), ok]() mutable { done(ok); });
        }
      }));

  for (int step = 0; step < kTotalSteps; ++step) {
    const InstanceKey step_key = StepKey(kPrefix, instance_key, step);
    const InstanceKey next_step_key = StepKey(kPrefix, instance_key, step + 1);
    const size_t split_id = SplitIndex(my_id, kGroupSize, step);
    llvm::StringRef split_data = GetSplit<T>(in_tensor_ref, kGroupSize,
                                             in_tensor.NumElements(), split_id);

    if (step == 0) {
      fabric_communicator->Send(
          next_step_key, neighbor_id, split_data,
          [refcounted_done = refcounted_done.CopyRef()](const bool ok) {
            refcounted_done->UpdateState(ok);
          });
    } else if (step <= kLastScatterStep) {
      // Scatter stage: send a chunk to the neighbor, aggregate the incoming
      // chunk with local buffer.
      callback_registry->SetCallback(
          step_key, [step, in_split = split_data, out_split = split_data,
                     reduction_fn, final_fn, fabric_communicator,
                     kLastScatterStep, kGroupSize, next_step_key, neighbor_id,
                     refcounted_done = refcounted_done.CopyRef()](
                        const InstanceKey&,
                        CallbackRegistry::CallbackValue callback_value) {
            // Scatter aggregates the results with the local buffer.
            reduction_fn(const_cast<char*>(callback_value->data()),
                         const_cast<char*>(in_split.data()), in_split.size());

            if (step == kLastScatterStep) {
              final_fn(const_cast<char*>(callback_value->data()),
                       in_split.size(), kGroupSize);
              std::copy(callback_value->begin(), callback_value->end(),
                        const_cast<char*>(out_split.begin()));
            }

            fabric_communicator->Send(
                next_step_key, neighbor_id,
                llvm::StringRef(callback_value->data(), callback_value->size()),
                [refcounted_done =
                     refcounted_done.CopyRef()](const bool ok) mutable {
                  refcounted_done->UpdateState(ok);
                });
          });
    } else {
      // Gather stage: an incoming chunk is final; just assign it to local
      // buffer and pass it to the neighbor as is.
      callback_registry->SetCallback(
          step_key,
          [step, out_split = split_data, fabric_communicator, kLastGatherStep,
           next_step_key, neighbor_id,
           refcounted_done = refcounted_done.CopyRef()](
              const InstanceKey&,
              CallbackRegistry::CallbackValue callback_value) mutable {
            // Gather assigns the incoming data to the local buffer
            std::copy(callback_value->begin(), callback_value->end(),
                      const_cast<char*>(out_split.begin()));
            if (step < kLastGatherStep) {
              fabric_communicator->Send(
                  next_step_key, neighbor_id,
                  llvm::StringRef(callback_value->data(),
                                  callback_value->size()),
                  [refcounted_done =
                       refcounted_done.CopyRef()](const bool ok) mutable {
                    refcounted_done->UpdateState(ok);
                  });
            }
          });
    }
  }
}

template <typename T>
void AllReduce(Argument<DistributedContext> dist_context,
               Argument<CollectiveGroup> collective_group,
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
  int my_index = FindMyIndex(collective_group.get(), dist_context->GetId());
  if (my_index == -1) {
    out_chain_indirect.SetError(
        "This worker is not part of the collective group ");
    return;
  }
  const HostId neighbor_id =
      collective_group
          ->members[(my_index + 1) % collective_group->members.size()];

  EnqueueWork(exec_ctx, [exec_ctx, instance_key = *instance_key,
                         dist_context = dist_context.ValueRef(),
                         collective_group = *collective_group,
                         in_tensor = in_tensor.ValueRef(),
                         out_tensor = out_tensor.ValueRef(), reduction_fn,
                         final_fn, my_id = dist_context->GetId(), neighbor_id,
                         out_chain = std::move(out_chain_indirect)] {
    DoAllReduce<T>(exec_ctx, dist_context.CopyRef(), instance_key,
                   collective_group, in_tensor.get(), out_tensor.get(),
                   reduction_fn, final_fn, my_id, neighbor_id,
                   out_chain.CopyRef());
  });
}

template <typename T>
void DoBroadcast(const InstanceKey& instance_key,
                 const CollectiveGroup& collective_group,
                 DenseHostTensor& tensor, HostId sender,
                 CallbackRegistry* registry,
                 FabricCommunicator* fabric_communicator, HostId my_id,
                 int my_index) {
  const auto kGroupSize = collective_group.members.size();
  const auto kNeighborId =
      collective_group.members[(my_index + 1) % kGroupSize];
  const auto kPrefix = collective_group.name;
  auto in_tensor = llvm::StringRef(reinterpret_cast<const char*>(tensor.data()),
                                   tensor.DataSizeInBytes());
  const auto num_elements = tensor.NumElements();
  auto chunks_collected = std::make_shared<std::atomic<int>>(0);
  const auto empty_callback_fn = [](const bool ok) {};

  for (auto i = 0; i < kGroupSize; ++i) {
    auto chunk_key = StepKey(kPrefix, instance_key, i);
    if (my_id == sender) {
      // A Sender sends data to its neighbor.
      auto payload = GetSplit<T>(in_tensor, kGroupSize, num_elements, i);
      fabric_communicator->Send(StepKey(kPrefix, chunk_key, kNeighborId),
                                kNeighborId, payload, empty_callback_fn);
    } else {
      auto recv_callback = [registry, instance_key, sender, chunk_key, i,
                            kPrefix, in_tensor, kGroupSize, kNeighborId,
                            chunks_collected, fabric_communicator, num_elements,
                            empty_callback_fn](
                               const InstanceKey&,
                               CallbackRegistry::CallbackValue data) mutable {
        // A neighbor receives data and forwards it to its neighbor.
        std::copy(
            data->begin(), data->end(),
            const_cast<char*>(
                GetSplit<T>(in_tensor, kGroupSize, num_elements, i).begin()));
        if (kNeighborId != sender) {
          fabric_communicator->Send(
              StepKey(kPrefix, chunk_key, kNeighborId), kNeighborId,
              llvm::StringRef(data->data(), data->size()), empty_callback_fn);
        }
        if (++(*chunks_collected) == kGroupSize) {
          registry->SetValue(instance_key, /*value=*/nullptr);
        }
      };
      registry->SetCallback(StepKey(kPrefix, chunk_key, my_id), recv_callback);
    }
  }
  if (my_id == sender) {
    registry->SetValue(instance_key, /*value=*/nullptr);
  }
}

template <typename T>
void Broadcast(Argument<DistributedContext> dist_context,
               Argument<CollectiveGroup> collective_group,
               Argument<InstanceKey> instance_key,
               Argument<DenseHostTensor> in_tensor, Argument<HostId> sender,
               Argument<Chain> in_chain, Result<Chain> out_chain,
               const ExecutionContext& exec_ctx) {
  auto out_chain_indirect = out_chain.Allocate();
  int my_index = FindMyIndex(collective_group.get(), dist_context->GetId());
  if (my_index == -1) {
    out_chain_indirect.SetError(
        "This worker is not part of the collective group ");
    return;
  }
  // Move AsyncValueRefs to on_done callback - this ensures that the arguments
  // will not be destroyed until the callback executes.
  auto on_done = [out_chain = std::move(out_chain_indirect),
                  in_tensor_ref = in_tensor.ValueRef(),
                  dist_context_ref = dist_context.ValueRef(),
                  instance_key = *instance_key](
                     const InstanceKey& broadcast_key,
                     CallbackRegistry::CallbackValue) {
    if (instance_key != broadcast_key) {
      out_chain.SetError("instance key mismatch in Broadcast callback");
    }
    out_chain.emplace();
  };
  CallbackRegistry* registry = dist_context->GetCallbackRegistry();
  registry->SetCallback(*instance_key, std::move(on_done));
  EnqueueWork(
      exec_ctx,
      [instance_key = *instance_key, collective_group = *collective_group,
       sender = *sender, registry,
       fabric_communicator = dist_context->GetOrCreateFabricCommunicator(),
       my_id = dist_context->GetId(), in_tensor_ref = in_tensor.ValueRef(),
       my_index] {
        DoBroadcast<T>(instance_key, collective_group, in_tensor_ref.get(),
                       sender, registry, fabric_communicator, my_id, my_index);
      });
}

AsyncValueRef<Chain> RemoteRegisterKernel(Chain ch,
                                          DistributedContext* dist_context,
                                          HostId receiver,
                                          StringAttribute program,
                                          StringAttribute program_name,
                                          const ExecutionContext& exec_ctx) {
  RemoteRegisterInvocation request;
  // program and program_name will live as long as out_chain is not populated.
  request.program = program.get();
  request.program_name = program_name.get();

  AsyncValueRef<Chain> out_chain =
      MakeConstructedAsyncValueRef<Chain>(exec_ctx.host());

  EnqueueWork(exec_ctx, [receiver, request, dist_context,
                         out_chain = out_chain.CopyRef()]() mutable {
    dist_context->GetOrCreateFabricCommunicator()->RemoteRegister(
        receiver, request,
        [out_chain = out_chain.CopyRef()](bool success) mutable {
          if (!success) {
            out_chain.SetError("Failed Remote Register");
          } else {
            out_chain.SetStateConcrete();
          }
        });
  });

  return out_chain;
}

AsyncValueRef<RemoteExecuteSpec> CreateRemoteExecuteSpec(
    RemainingArguments inputs, const ExecutionContext& exec_ctx) {
  AsyncValueRef<RemoteExecuteSpec> value =
      MakeAvailableAsyncValueRef<RemoteExecuteSpec>(exec_ctx.host());
  value->output_devices.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    const std::string& device_str = inputs[i]->get<std::string>();
    RCReference<Device> device =
        exec_ctx.host()->GetDeviceManager()->GetDeviceRef<Device>(device_str);
    if (device.get() == nullptr) {
      TFRT_LOG(ERROR) << "Can't find device: " << device_str;
      return MakeErrorAsyncValueRef(exec_ctx.host(),
                                    StrCat("Can't find device: ", device_str));
    }
    value->output_devices.push_back(device.CopyRef());
  }
  return value;
}

void RemoteExecuteKernel(Chain ch, DistributedContext* dist_context,
                         HostId receiver, Argument<RemoteExecuteSpec> spec,
                         RemainingArguments inputs, RemainingResults results,
                         StringAttribute program_name,
                         const ExecutionContext& exec_ctx) {
  RemoteExecuteInvocation request;
  // program_name will live as long as out_chain is not populated.
  request.program_name = program_name.get();

  request.inputs.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    const RemoteObjectId& input = inputs[i]->get<RemoteObjectId>();
    request.inputs.emplace_back(input.prefix_id, input.local_id,
                                input.device->name());
  }

  AsyncValueRef<Chain> out_chain =
      MakeConstructedAsyncValueRef<Chain>(exec_ctx.host());
  results[0] = out_chain.CopyRef();

  RemoteObjectManager* manager = dist_context->GetRemoteObjectManager();
  if (spec->output_devices.size() != results.size() - 1) {
    out_chain.SetError(
        StrCat("Mismatch output devices size in RemoteExecuteSpec: ",
               spec->output_devices.size(), " expected: ", results.size() - 1));
    return;
  }
  request.outputs.reserve(results.size() - 1);
  for (int i = 1; i < results.size(); ++i) {
    AsyncValueRef<RemoteObjectId> out_id =
        MakeAvailableAsyncValueRef<RemoteObjectId>(
            exec_ctx.host(), manager->AllocateRemoteObject(
                                 spec->output_devices[i - 1].CopyRef()));
    request.outputs.emplace_back(out_id->prefix_id, out_id->local_id,
                                 out_id->device->name());
    results[i] = out_id.CopyRef();
  }

  EnqueueWork(exec_ctx,
              [receiver, request, dist_context, out_chain = out_chain.CopyRef(),
               spec = spec.ValueRef()]() mutable {
                dist_context->GetOrCreateFabricCommunicator()->RemoteExecute(
                    receiver, request,
                    [out_chain = out_chain.CopyRef()](bool success) mutable {
                      if (!success) {
                        out_chain.SetError("Failed Remote Execute");
                      } else {
                        out_chain.SetStateConcrete();
                      }
                    });
              });
}
}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
void RegisterDistributedKernels(KernelRegistry* registry) {
  registry->AddKernel("dist.create_collective_group",
                      TFRT_KERNEL(CreateCollectiveGroup));
  registry->AddKernel("dist.cpu.allreduce.f32", TFRT_KERNEL(AllReduce<float>));
  registry->AddKernel("dist.cpu.allreduce.i32",
                      TFRT_KERNEL(AllReduce<int32_t>));
  registry->AddKernel("dist.cpu.broadcast.f32", TFRT_KERNEL(Broadcast<float>));
  registry->AddKernel("dist.cpu.broadcast.i32",
                      TFRT_KERNEL(Broadcast<int32_t>));
  registry->AddKernel("dist.create_remote_execute_spec",
                      TFRT_KERNEL(CreateRemoteExecuteSpec));
  registry->AddKernel("dist.remote_execute", TFRT_KERNEL(RemoteExecuteKernel));
  registry->AddKernel("dist.remote_register",
                      TFRT_KERNEL(RemoteRegisterKernel));
}

}  // namespace tfrt
