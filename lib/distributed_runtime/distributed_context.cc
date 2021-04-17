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

//===- distributed_context.cc - Distributed Context ------*- C++ -*--------===//
//
// Contains implementation of DistributedContext class.

#include "tfrt/distributed_runtime/distributed_context.h"

#include "llvm/ADT/DenseMap.h"
#include "tfrt/bef_converter/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/cluster_info.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/function_cache.h"
#include "tfrt/distributed_runtime/proto/remote_message.pb.h"
#include "tfrt/distributed_runtime/remote_client.h"
#include "tfrt/distributed_runtime/remote_device.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/distributed_runtime/task_name_util.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/refcounted_callback.h"

namespace tfrt {

DistributedContext::DistributedContext(
    uint64_t context_id, ServerContext* server_context,
    DistributedContextConfiguration configuration)
    : context_id_(context_id),
      server_context_(server_context),
      dist_config_(configuration),
      cluster_info_(configuration),
      collective_groups_(InitializeCollectiveGroups(configuration)),
      remote_manager_(std::make_unique<RemoteObjectManager>(
          cluster_info_.GetTaskHandle(), GetHostContext())),
      callback_registry_(new CallbackRegistry()),
      function_cache_(new FunctionCache(GetHostContext())) {
  TaskHandle task_handle = cluster_info_.GetTaskHandle();
  DeviceManager* local_device_mgr = GetHostContext()->GetDeviceManager();
  // For each local device, add a corresponding RemoveDevice instance with the
  // the fully specified device name to the remote device manager.
  for (auto& device : local_device_mgr->ListDevices<Device>()) {
    const std::string& device_name = TaskNameUtil::ConcatDeviceName(
        dist_config_.job_name(), dist_config_.task_id(),
        TaskNameUtil::StripDevicePrefix(device->name()));
    auto remote_device =
        NewRemoteDevice(device_name, device->type().name(), task_handle);
    TFRT_DLOG_IF(FATAL, !remote_device) << StrCat(remote_device.takeError());
    cluster_device_mgr_.MaybeAddDevice(TakeRef(remote_device.get()));
  }

  // Create remote chain object for current task in the DistributedContext.
  const std::string& host_cpu_device_name = TaskNameUtil::ConcatDeviceName(
      dist_config_.job_name(), dist_config_.task_id(),
      HostContext::kDefaultHostDeviceName);
  RCReference<Device> cpu_device = cluster_device_mgr_.MaybeAddDevice(
      MakeRef<RemoteCpuDevice>(host_cpu_device_name, task_handle));
  assert(cpu_device);
  if (cpu_device.get() != nullptr) {
    local_ready_chain_ = std::make_unique<RemoteObjectId>(
        remote_manager_->AllocateRemoteObject(cpu_device.CopyRef()));
    remote_manager_->SetRemoteObject(
        *local_ready_chain_, GetReadyChain(GetHostContext()).CopyRCRef());
  }
}

DistributedContext::~DistributedContext() {}

llvm::StringMap<CollectiveGroup> DistributedContext::InitializeCollectiveGroups(
    const DistributedContextConfiguration& config) {
  llvm::StringMap<CollectiveGroup> collective_groups;
  for (const auto& group_config : config.collective_groups()) {
    llvm::SmallVector<TaskHandle, 8> members;
    members.reserve(group_config.members_size());
    for (const auto& task : group_config.members()) {
      members.push_back(cluster_info_.GetTaskHandle(task).get());
    }
    collective_groups.try_emplace(
        group_config.name(), CollectiveGroup{group_config.name(), members});
  }
  return collective_groups;
}

const CollectiveGroup& DistributedContext::GetCollectiveGroup(
    string_view name) const {
  const auto& it = collective_groups_.find(name);
  assert(it != collective_groups_.end() && "Failed to find collective group.");
  return it->second;
}

RemoteClientInterface* DistributedContext::GetRemoteClient(
    TaskHandle task_handle) {
  mutex_lock l(remote_clients_mu_);
  auto it = remote_clients_.find(task_handle);
  if (it == remote_clients_.end()) {
    auto* communicator = server_context_->GetOrCreateFabricCommunicator();
    auto ret = remote_clients_.try_emplace(
        task_handle, communicator->CreateRemoteClient(this, task_handle));
    assert(ret.second && "Failed to create remote client.");
    it = ret.first;
  }
  return it->second.get();
}

void DistributedContext::GetRemoteDevices(
    DistributedContext::CallbackFn done_callback) {
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = MakeRef<RefCountedCallback>(std::move(done_callback));

  auto request = std::make_shared<GetDevicesRequest>();
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      // As the first round of RPCs for cluster initialization, the GetDevices
      // RPCs are given a large limit of max_retries to allow remote workers
      // to start after the client.
      RemoteCallContext call_ctx(/*max_retries=*/64, /*timeout_ms=*/0);
      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_unique<GetDevicesResponse>();
      client->GetDevicesAsync(
          &call_ctx, request.get(), response.get(),
          [request, response = std::move(response), rc_done = rc_done.CopyRef(),
           this, job_name = job_config.name(), task_id = task.first,
           task_handle](Error e) mutable {
            if (e) {
              rc_done->UpdateState(std::move(e));
              return;
            }
            for (const auto& device_info : response->devices()) {
              const std::string device_name = TaskNameUtil::ConcatDeviceName(
                  job_name, task_id,
                  TaskNameUtil::StripDevicePrefix(device_info.name()));
              auto expected =
                  NewRemoteDevice(device_name, device_info.type(), task_handle);
              if (expected) {
                cluster_device_mgr_.MaybeAddDevice(TakeRef(expected.get()));
              } else {
                rc_done->UpdateState(expected.takeError());
              }
            }
          });
    }
  }
}

void DistributedContext::CreateRemoteContexts(
    RemoteInitMode mode, DistributedContext::CallbackFn done_callback) {
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = MakeRef<RefCountedCallback>(
      [this, mode, done = std::move(done_callback)](Error e) mutable {
        // Only send keep-alive to prevent remote context timeout if not created
        // in multi-client mode (otherwise, the context is owned by the remote
        // client and will not be subject to garbage collection).
        if (mode == RemoteInitMode::SINGLE_CLIENT) {
          SendKeepAlive(
              server_context_->GetConfiguration().context_gc_timeout_secs / 2);
        }
        done(std::move(e));
      });

  // Base request contains information that is the shared by all the requests
  // sent to different tasks, including the cluster configuration and collective
  // groups of distributed context configuration.
  // Individual requests can directly set to use the allocated fields of the
  // base one without memory copies. The base request must be alive until all
  // uses of individual requests have finished.
  auto base_request = std::make_shared<CreateContextRequest>();
  *base_request->mutable_dist_config()->mutable_cluster_config() =
      dist_config_.cluster_config();
  *base_request->mutable_dist_config()->mutable_collective_groups() =
      dist_config_.collective_groups();
  for (const auto& device : cluster_device_mgr_.ListDevices<Device>()) {
    auto device_info = base_request->add_devices();
    device_info->set_name(device->name().str());
    device_info->set_type(device->type().name().str());
  }
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      if (job_config.name() == dist_config_.job_name() &&
          task.first == dist_config_.task_id()) {
        continue;
      }

      auto request = std::make_unique<CreateContextRequest>();
      request->set_context_id(context_id_);
      auto* request_dist_config = request->mutable_dist_config();
      request_dist_config->set_job_name(job_config.name());
      request_dist_config->set_task_id(task.first);
      request_dist_config->unsafe_arena_set_allocated_cluster_config(
          base_request->mutable_dist_config()->mutable_cluster_config());
      for (auto& cg :
           *base_request->mutable_dist_config()->mutable_collective_groups()) {
        request_dist_config->mutable_collective_groups()
            ->UnsafeArenaAddAllocated(&cg);
      }
      for (auto& device : *base_request->mutable_devices()) {
        request->mutable_devices()->UnsafeArenaAddAllocated(&device);
      }
      request->set_is_multi_client(mode == RemoteInitMode::MULTI_CLIENT);

      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_unique<CreateContextResponse>();
      client->CreateContextAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [this, task_handle, base_request, request = std::move(request),
           response = std::move(response),
           rc_done = rc_done.CopyRef()](Error e) mutable {
            if (e) {
              rc_done->UpdateState(std::move(e));
            } else {
              rc_done->UpdateState(
                  AddReadyChain(task_handle, response->ready_chain()));
            }
            // NOTE: `base_request` is the owner of `cluster_config` and
            // `collective_groups`. Release these fields from `request` so that
            // these fields are not destructed multiple times.
            auto request_dist_config = request->mutable_dist_config();
            request_dist_config->unsafe_arena_release_cluster_config();
            const int n_groups = request_dist_config->collective_groups_size();
            for (int i = 0; i < n_groups; i++) {
              request_dist_config->mutable_collective_groups()
                  ->UnsafeArenaReleaseLast();
            }
            const int n_devices = request->devices_size();
            for (int i = 0; i < n_devices; i++) {
              request->mutable_devices()->UnsafeArenaReleaseLast();
            }
          });
    }
  }
}

Error DistributedContext::AddReadyChain(TaskHandle task_handle,
                                        const RemoteObjectIdProto& chain) {
  RCReference<Device> device =
      cluster_device_mgr_.GetDeviceRef<Device>(chain.device());
  if (device.get() == nullptr) {
    return llvm::make_error<DeviceNotFoundErrorInfo>(
        StrCat("Can't find device: ", chain.device()));
  }
  RemoteObjectId ready_chain(chain.prefix_id(), chain.local_id(),
                             device.CopyRef());
  mutex_lock l(ready_chains_mu_);
  ready_chains_.insert({task_handle, ready_chain});
  return Error::success();
}

namespace {
void RemoteObjectIdToProto(RemoteObjectId& obj_id, RemoteObjectIdProto* proto) {
  proto->set_device(obj_id.device->name().str());
  proto->set_prefix_id(obj_id.prefix_id);
  proto->set_local_id(obj_id.local_id);
}
}  // namespace

void DistributedContext::BroadcastRemoteReadyChains(
    DistributedContext::CallbackFn done_callback) {
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = MakeRef<RefCountedCallback>(std::move(done_callback));
  // Create a copy of remote ready chains to avoid frequent mutex locking when
  // constructng the reqeust.
  auto ready_chain_objs = RemoteReadyChains();

  auto request = std::make_shared<SendReadyChainsRequest>();
  request->set_context_id(context_id_);
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      auto ready_chain = request->add_ready_chains();
      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      if (task_handle == GetTaskHandle()) {
        RemoteObjectIdToProto(*local_ready_chain_, ready_chain);
      } else {
        auto it = ready_chain_objs.find(task_handle);
        if (it == ready_chain_objs.end()) {
          rc_done->UpdateState(llvm::make_error<UnknownErrorInfo>(StrCat(
              "Missing remote ready chain from ",
              TaskNameUtil::ConcatTaskName(job_config.name(), task.first))));
        } else {
          RemoteObjectIdToProto(it->getSecond(), ready_chain);
        }
      }
    }
  }

  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      if (job_config.name() == dist_config_.job_name() &&
          task.first == dist_config_.task_id()) {
        continue;
      }
      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_shared<SendReadyChainsResponse>();
      client->SendReadyChainsAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request, response, rc_done = rc_done.CopyRef()](Error e) mutable {
            rc_done->UpdateState(std::move(e));
          });
    }
  }
}

void DistributedContext::CloseRemoteContexts(
    DistributedContext::CallbackFn done_callback) {
  {
    mutex_lock l(keep_alive_mu_);
    if (keep_alive_timer_.get() != nullptr) {
      GetHostContext()->GetTimerQueue()->CancelTimer(keep_alive_timer_);
    }
  }
  // Reference-counted done callback is invoked after all remote calls finish
  auto rc_done = MakeRef<RefCountedCallback>(std::move(done_callback));

  auto request = std::make_shared<CloseContextRequest>();
  request->set_context_id(context_id_);
  for (const auto& job_config : dist_config_.cluster_config().jobs()) {
    for (const auto& task : job_config.tasks()) {
      if (job_config.name() == dist_config_.job_name() &&
          task.first == dist_config_.task_id()) {
        continue;
      }
      TaskHandle task_handle = GetTaskHandle(job_config.name(), task.first);
      RemoteClientInterface* client = GetRemoteClient(task_handle);
      auto response = std::make_shared<CloseContextResponse>();
      client->CloseContextAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request, response, rc_done = rc_done.CopyRef()](Error e) mutable {
            rc_done->UpdateState(std::move(e));
          });
    }
  }
}

llvm::DenseMap<TaskHandle, RemoteObjectId>
DistributedContext::RemoteReadyChains() {
  mutex_lock l(ready_chains_mu_);
  return ready_chains_;
}

void DistributedContext::SendKeepAlive(int delay_secs) {
  auto keep_alive_fn = [this, delay_secs]() mutable {
    auto done = MakeRef<RefCountedCallback>([this, delay_secs](Error e) {
      if (e) {
        TFRT_LOG(ERROR) << "Error in DistributedContext::SendKeepAlive: "
                        << StrCat(e);
      }
      SendKeepAlive(delay_secs);
    });
    auto request = std::make_shared<KeepAliveRequest>();
    request->set_context_id(context_id_);
    auto response = std::make_shared<KeepAliveResponse>();

    mutex_lock l(remote_clients_mu_);
    for (const auto& pair : remote_clients_) {
      pair.second->KeepAliveAsync(
          RemoteCallContext::GetDefault(), request.get(), response.get(),
          [request, response, done = done.CopyRef()](Error e) {
            done->UpdateState(std::move(e));
          });
    }
  };
  mutex_lock l(keep_alive_mu_);
  keep_alive_timer_ = GetHostContext()->GetTimerQueue()->ScheduleTimer(
      std::chrono::seconds(delay_secs), keep_alive_fn);
}

}  // namespace tfrt
