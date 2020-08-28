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
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/logging.h"
#include "tfrt/tensor/dense_host_tensor.h"

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

llvm::StringRef GetSplit(llvm::StringRef split_input, int num_splits,
                         size_t dtype_size, int split_id) {
  const size_t num_elements = split_input.size() / dtype_size;
  if (num_elements < num_splits) {
    TFRT_LOG(FATAL)
        << "Expected at least one element per split for num_elements="
        << num_elements << " and num_splits=" << num_splits;
  }
  const size_t num_elements_per_split = std::floor(num_elements / num_splits);
  const size_t data_size_per_split = num_elements_per_split * dtype_size;
  if (split_id * data_size_per_split > split_input.size()) {
    TFRT_LOG(FATAL) << "The given split id does not exist.";
  }
  const char* data_ptr = split_input.data();
  size_t split_size = data_size_per_split;
  // The last split gets leftover.
  if (split_id * data_size_per_split + data_size_per_split >
      split_input.size()) {
    split_size = split_input.size() - (split_id * data_size_per_split);
  }
  return llvm::StringRef(data_ptr + split_id * data_size_per_split, split_size);
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
  int my_index = -1;
  for (auto i = 0; i < collective_group.members.size(); ++i) {
    if (collective_group.members[i] == my_id) {
      my_index = i;
    }
  }
  return my_index;
}

CollectiveGroup CreateCollectiveGroup(Argument<DistributedContext> dist_context,
                                      Argument<std::string> name,
                                      const ExecutionContext& exec_ctx) {
  auto collective_group = dist_context->GetCollectiveGroup(name.get());
  return collective_group;
}

void DoAllReduce(const InstanceKey& instance_key,
                 const CollectiveGroup& collective_group, size_t dtype_size,
                 const DenseHostTensor& in_tensor,
                 const DenseHostTensor& out_tensor,
                 ElementWiseReductionFunction reduction_fn,
                 ElementWiseFinalFunction final_fn,
                 CallbackRegistry* callback_registry,
                 FabricCommunicator* fabric_communicator, HostId my_id,
                 int my_index) {
  const auto kGroupSize = collective_group.members.size();
  const auto kNeighborId =
      collective_group.members[(my_index + 1) % kGroupSize];
  const auto kLastScatterStep = kGroupSize - 1;
  const auto kLastGatherStep = 2 * kGroupSize - 2;
  const auto kTotalGatherSteps = kGroupSize - 1;
  const auto kPrefix = collective_group.name;
  auto in_tensor_ref =
      llvm::StringRef(reinterpret_cast<const char*>(in_tensor.data()),
                      in_tensor.DataSizeInBytes());
  auto counter = std::make_shared<std::atomic<int>>(0);

  for (int step = 0; step <= 2 * kGroupSize - 2; ++step) {
    auto step_key = StepKey(kPrefix, instance_key, step);

    if (step == 0) {
      llvm::StringRef payload = GetSplit(in_tensor_ref, kGroupSize, dtype_size,
                                         SplitIndex(my_id, kGroupSize, step));
      fabric_communicator->Send(StepKey(kPrefix, instance_key, 1), kNeighborId,
                                payload);
    } else if (step <= kLastScatterStep) {
      // Scatter Stage: Send a chunk to the neighbor, aggregate the incoming
      // chunk with local buffer.
      auto recv_callback = [step, instance_key,
                            in_split =
                                GetSplit(in_tensor_ref, kGroupSize, dtype_size,
                                         SplitIndex(my_id, kGroupSize, step)),
                            out_split =
                                GetSplit(in_tensor_ref, kGroupSize, dtype_size,
                                         SplitIndex(my_id, kGroupSize, step)),
                            reduction_fn, final_fn, fabric_communicator,
                            kLastScatterStep, kGroupSize, kPrefix, kNeighborId](
                               const InstanceKey&,
                               CallbackRegistry::CallbackValue callback_value) {
        // Scatter aggregates the results with the local buffer.
        reduction_fn(const_cast<char*>(callback_value->data()),
                     const_cast<char*>(in_split.data()), in_split.size());

        if (step == kLastScatterStep) {
          final_fn(const_cast<char*>(callback_value->data()), in_split.size(),
                   kGroupSize);
          std::copy(callback_value->begin(), callback_value->end(),
                    const_cast<char*>(out_split.begin()));
        }

        fabric_communicator->Send(
            StepKey(kPrefix, instance_key, step + 1), kNeighborId,
            llvm::StringRef(callback_value->data(), callback_value->size()));
      };
      callback_registry->SetCallback(step_key, recv_callback);
    } else {
      // Gather Stage: An incoming chunk is final. Just assign it to local
      // buffer and pass it to the neighbour as is.
      auto recv_callback =
          [step, counter, instance_key,
           out_split = GetSplit(in_tensor_ref, kGroupSize, dtype_size,
                                SplitIndex(my_id, kGroupSize, step)),
           fabric_communicator, callback_registry, kLastGatherStep, kPrefix,
           kNeighborId, kTotalGatherSteps](
              const InstanceKey&,
              CallbackRegistry::CallbackValue callback_value) mutable {
            // Gather assigns the incoming data to the local buffer
            std::copy(callback_value->begin(), callback_value->end(),
                      const_cast<char*>(out_split.begin()));
            if (step < kLastGatherStep) {
              fabric_communicator->Send(
                  StepKey(kPrefix, instance_key, step + 1), kNeighborId,
                  llvm::StringRef(callback_value->data(),
                                  callback_value->size()));
            }
            if (++(*counter) == kTotalGatherSteps) {
              callback_registry->SetValue(instance_key, nullptr);
            }
          };
      callback_registry->SetCallback(step_key, recv_callback);
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

  // Move AsyncValueRefs to on_done callback - this ensures that the arguments
  // will not be destroyed until the callback executes.
  auto on_done =
      [dist_context = dist_context.ValueRef(), instance_key = *instance_key,
       in_tensor = in_tensor.ValueRef(), out_tensor = out_tensor.ValueRef(),
       out_chain = std::move(out_chain_indirect)](
          const InstanceKey& reduced_key, CallbackRegistry::CallbackValue) {
        if (instance_key != reduced_key) {
          out_chain.SetError("instance key mismatch in AllReduce callback");
        }
        out_chain.emplace();
      };
  CallbackRegistry* callback_registry = dist_context->GetCallbackRegistry();
  callback_registry->SetCallback(*instance_key, std::move(on_done));
  exec_ctx.host()->EnqueueWork(
      [instance_key = *instance_key, collective_group = *collective_group,
       dtype_size = sizeof(T), in_tensor = in_tensor.ValueRef(),
       out_tensor = out_tensor.ValueRef(), reduction_fn, final_fn,
       callback_registry = dist_context->GetCallbackRegistry(),
       fabric_communicator = dist_context->GetOrCreateFabricCommunicator(),
       my_id = dist_context->GetId(), my_index] {
        DoAllReduce(instance_key, collective_group, dtype_size, in_tensor.get(),
                    out_tensor.get(), reduction_fn, final_fn, callback_registry,
                    fabric_communicator, my_id, my_index);
      });
}

void DoBroadcast(const InstanceKey& instance_key,
                 const CollectiveGroup& collective_group, size_t dtype_size,
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
  auto chunks_collected = std::make_shared<std::atomic<int>>(0);

  for (auto i = 0; i < kGroupSize; ++i) {
    auto chunk_key = StepKey(kPrefix, instance_key, i);
    if (my_id == sender) {
      // A Sender sends data to its neighbor.
      auto payload = GetSplit(in_tensor, kGroupSize, dtype_size, i);
      fabric_communicator->Send(StepKey(kPrefix, chunk_key, kNeighborId),
                                kNeighborId, payload);
    } else {
      auto recv_callback = [registry, instance_key, sender, chunk_key, i,
                            kPrefix, in_tensor, kGroupSize, kNeighborId,
                            dtype_size, chunks_collected, fabric_communicator](
                               const InstanceKey&,
                               CallbackRegistry::CallbackValue data) mutable {
        // A neighbor receives data and forwards it to its neighbor.
        std::copy(data->begin(), data->end(),
                  const_cast<char*>(
                      GetSplit(in_tensor, kGroupSize, dtype_size, i).begin()));
        if (kNeighborId != sender) {
          fabric_communicator->Send(
              StepKey(kPrefix, chunk_key, kNeighborId), kNeighborId,
              llvm::StringRef(data->data(), data->size()));
        }
        if (++(*chunks_collected) == kGroupSize) {
          registry->SetValue(instance_key, nullptr);
        }
      };
      registry->SetCallback(StepKey(kPrefix, chunk_key, my_id), recv_callback);
    }
  }
  if (my_id == sender) {
    registry->SetValue(instance_key, nullptr);
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
  exec_ctx.host()->EnqueueWork(
      [instance_key = *instance_key, collective_group = *collective_group,
       dtype_size = sizeof(T), sender = *sender, registry,
       fabric_communicator = dist_context->GetOrCreateFabricCommunicator(),
       my_id = dist_context->GetId(), in_tensor_ref = in_tensor.ValueRef(),
       my_index] {
        DoBroadcast(instance_key, collective_group, dtype_size,
                    in_tensor_ref.get(), sender, registry, fabric_communicator,
                    my_id, my_index);
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
}

}  // namespace tfrt
