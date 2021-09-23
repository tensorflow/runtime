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

// This file defines tensor handle.

#include "tfrt/core_runtime/tensor_handle.h"

#include "llvm/Support/raw_ostream.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/device.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {

namespace {

// Infer the destination tensor type for TransferTo. Input device is the
// destination device, and input tensor is the source tensor.
inline TensorType InferDstTensorTypeFromDevice(const Device& device,
                                               const Tensor& tensor) {
  TensorType dst_tensor_type = tensor.tensor_type();
  // If the source tensor is not on host, and the destination device is host,
  // it must be a dense tensor to be transferred from device.
  if (device.type() == GetStaticDeviceType("cpu") && !tensor.IsHostTensor()) {
    dst_tensor_type = tfrt::GetStaticTensorType("DenseHost");
  } else if (device.type() == GetStaticDeviceType("gpu")) {
    dst_tensor_type = tfrt::GetStaticTensorType("DenseGpu");
  } else if (device.type() == GetStaticDeviceType("tpu")) {
    dst_tensor_type = tfrt::GetStaticTensorType("DenseTpu");
  }
  return dst_tensor_type;
}

}  // namespace

TensorHandle::TensorHandle(AsyncValueRef<RCReference<Device>> async_device,
                           AsyncValueRef<TensorMetadata> async_metadata,
                           AsyncValueRef<Tensor> tensor) {
  assert(async_metadata.GetAsyncValue());
  assert(tensor.GetAsyncValue());
  assert(async_device.GetAsyncValue());
  uint32_t flags = 0;

  RCReference<AsyncValue> error;
  if (async_metadata.IsError()) {
    error = async_metadata.CopyRCRef();
  } else if (tensor.IsError()) {
    error = tensor.CopyRCRef();
  } else if (async_device.IsError()) {
    error = async_device.CopyRCRef();
  }

  if (error) {
    tensor_and_flags_.setPointerAndInt(error.CopyRef().release(), flags);
    new (&async_device_) AsyncValueRef<RCReference<Device>>(error);
    new (&async_metadata_) AsyncValueRef<TensorMetadata>(std::move(error));
  } else {
    tensor_and_flags_.setPointerAndInt(tensor.release(), flags);
    new (&async_device_)
        AsyncValueRef<RCReference<Device>>(std::move(async_device));
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(std::move(async_metadata));
  }
}

TensorHandle::TensorHandle(AsyncValueRef<RCReference<Device>> async_device,
                           const TensorMetadata& metadata,
                           AsyncValueRef<Tensor> tensor) {
  assert(tensor.GetAsyncValue());
  assert(async_device.GetAsyncValue());
  uint32_t flags = Flags::MetadataInline;

  RCReference<AsyncValue> error;
  if (tensor.IsError()) {
    error = tensor.CopyRCRef();
  } else if (async_device.IsError()) {
    error = async_device.CopyRCRef();
  }

  if (error) {
    tensor_and_flags_.setPointerAndInt(error.CopyRef().release(), flags);
    new (&async_device_) AsyncValueRef<RCReference<Device>>(std::move(error));

  } else {
    tensor_and_flags_.setPointerAndInt(tensor.release(), flags);
    new (&async_device_)
        AsyncValueRef<RCReference<Device>>(std::move(async_device));
  }

  new (&inlined_metadata_) TensorMetadata(metadata);
}

TensorHandle::TensorHandle(RCReference<Device> device,
                           AsyncValueRef<TensorMetadata> async_metadata,
                           AsyncValueRef<Tensor> tensor) {
  assert(async_metadata.GetAsyncValue());
  assert(tensor.GetAsyncValue());
  assert(device);
  uint32_t flags = Flags::DeviceInline;

  RCReference<AsyncValue> error;
  if (tensor.IsError()) {
    error = tensor.CopyRCRef();
  } else if (async_metadata.IsError()) {
    error = async_metadata.CopyRCRef();
  }

  if (error) {
    tensor_and_flags_.setPointerAndInt(error.CopyRef().release(), flags);
    new (&async_metadata_) AsyncValueRef<TensorMetadata>(std::move(error));
  } else {
    tensor_and_flags_.setPointerAndInt(tensor.release(), flags);
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(std::move(async_metadata));
  }
  new (&inlined_device_) RCReference<Device>(std::move(device));
}

TensorHandle::TensorHandle(RCReference<Device> device,
                           const TensorMetadata& metadata,
                           AsyncValueRef<Tensor> tensor) {
  assert(tensor.GetAsyncValue());
  assert(device);
  uint32_t flags = Flags::DeviceInline | Flags::MetadataInline;

  tensor_and_flags_.setPointerAndInt(tensor.release(), flags);
  new (&inlined_metadata_) TensorMetadata(metadata);
  new (&inlined_device_) RCReference<Device>(std::move(device));
}

TensorHandle::TensorHandle(AsyncValueRef<TensorHandle> error) {
  assert(error.IsError());
  tensor_and_flags_.setPointerAndInt(error.CopyRef().release(), 0);
  new (&async_device_) AsyncValueRef<RCReference<Device>>(error.CopyRef());
  new (&async_metadata_) AsyncValueRef<TensorMetadata>(std::move(error));
}

TensorHandle TensorHandle::CreateError(RCReference<AsyncValue> error) {
  assert(error->IsError());
  auto th = AsyncValueRef<TensorHandle>(std::move(error));
  return TensorHandle(std::move(th));
}

ErrorAsyncValue* TensorHandle::GetErrorAsyncValue() {
  assert(
      IsError() &&
      "Cannot call GetErrorAsyncValue() if it is not an error TensorHandle.");
  if (GetAsyncTensor()->IsError()) {
    return llvm::cast<ErrorAsyncValue>(GetAsyncTensor());
  }
  return llvm::cast<ErrorAsyncValue>(async_metadata_.GetAsyncValue());
}

TensorHandle TensorHandle::TransferTo(const ExecutionContext& exec_ctx,
                                      RCReference<Device> dst,
                                      TensorType dst_tensor_type) const {
  HostContext* host = exec_ctx.host();
  AsyncValueRef<Tensor> result_tensor;
  if (GetAsyncTensor()->IsError()) {
    return TensorHandle(AsyncValueRef<TensorHandle>(FormRef(GetAsyncTensor())));
  }
  if (GetAsyncTensor()->IsAvailable() && IsDeviceAvailable()) {
    const Device& src = *GetAvailableDevice();
    auto& tensor = GetAsyncTensor()->get<Tensor>();
    if (dst.get() == &src && tensor.IsTensorType(dst_tensor_type))
      return CopyRef();
    result_tensor = ConvertTensor(exec_ctx, tensor, src, *dst, dst_tensor_type);
  } else {
    RCReference<IndirectAsyncValue> result_ind_av =
        MakeIndirectAsyncValue(host);
    result_tensor = AsyncValueRef<Tensor>(result_ind_av);
    SmallVector<AsyncValue*, 2> async_values;
    async_values.push_back(GetAsyncTensor());
    if (!IsDeviceAvailable()) {
      async_values.push_back(GetAsyncDevice().GetAsyncValue());
    }
    RunWhenReady(async_values, [th = CopyRef(),
                                result_ind_av = std::move(result_ind_av), dst,
                                dst_tensor_type, exec_ctx]() {
      if (th.IsDeviceError()) {
        result_ind_av->ForwardTo(th.GetAsyncDevice().CopyRCRef());
        return;
      }
      if (th.GetAsyncTensor()->IsError()) {
        result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
        return;
      }
      auto& tensor = th.GetAsyncTensor()->get<Tensor>();
      if (dst.get() == th.GetAvailableDevice().get() &&
          tensor.IsTensorType(dst_tensor_type)) {
        result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
      } else {
        result_ind_av->ForwardTo(ConvertTensor(
            exec_ctx, tensor, *th.GetAvailableDevice(), *dst, dst_tensor_type));
      }
    });
  }

  if (IsMetadataAvailable()) {
    return TensorHandle(std::move(dst), GetAvailableMetadata(),
                        std::move(result_tensor));
  } else {
    return TensorHandle(std::move(dst), GetAsyncMetadata().CopyRef(),
                        std::move(result_tensor));
  }
}

TensorHandle TensorHandle::TransferToSameDevice(
    const ExecutionContext& exec_ctx, TensorType dst_tensor_type) const {
  if (IsDeviceAvailable()) {
    return TransferTo(exec_ctx, GetAvailableDevice(), dst_tensor_type);
  }
  RCReference<IndirectAsyncValue> result_ind_av =
      MakeIndirectAsyncValue(exec_ctx.host());
  AsyncValueRef<Tensor> result_tensor = AsyncValueRef<Tensor>(result_ind_av);
  SmallVector<AsyncValue*, 2> async_values;
  async_values.push_back(GetAsyncTensor());
  async_values.push_back(GetAsyncDevice().GetAsyncValue());
  RunWhenReady(
      async_values, [th = CopyRef(), result_ind_av = std::move(result_ind_av),
                     dst_tensor_type, exec_ctx]() {
        if (th.IsDeviceError()) {
          result_ind_av->ForwardTo(th.GetAsyncDevice().CopyRCRef());
          return;
        }
        if (th.GetAsyncTensor()->IsError()) {
          result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
          return;
        }
        auto& tensor = th.GetAsyncTensor()->get<Tensor>();
        if (tensor.IsTensorType(dst_tensor_type)) {
          result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
        } else {
          result_ind_av->ForwardTo(
              ConvertTensor(exec_ctx, tensor, *th.GetAvailableDevice(),
                            *th.GetAvailableDevice(), dst_tensor_type));
        }
      });
  if (IsMetadataAvailable()) {
    return TensorHandle(GetAsyncDevice().CopyRef(), GetAvailableMetadata(),
                        std::move(result_tensor));
  } else {
    return TensorHandle(GetAsyncDevice().CopyRef(),
                        GetAsyncMetadata().CopyRef(), std::move(result_tensor));
  }
}

TensorHandle TensorHandle::TransferToInferredType(
    const ExecutionContext& exec_ctx, RCReference<Device> dst) const {
  HostContext* host = exec_ctx.host();
  AsyncValueRef<Tensor> result_tensor;

  if (GetAsyncTensor()->IsError()) {
    return TensorHandle(AsyncValueRef<TensorHandle>(FormRef(GetAsyncTensor())));
  }
  if (GetAsyncTensor()->IsAvailable() && IsDeviceAvailable()) {
    const Device& src = *GetAvailableDevice();
    auto& tensor = GetAsyncTensor()->get<Tensor>();
    if (dst.get() == &src) return CopyRef();
    TensorType dst_tensor_type = InferDstTensorTypeFromDevice(*dst, tensor);
    result_tensor = ConvertTensor(exec_ctx, tensor, src, *dst, dst_tensor_type);
  } else {
    RCReference<IndirectAsyncValue> result_ind_av =
        MakeIndirectAsyncValue(host);
    result_tensor = AsyncValueRef<Tensor>(result_ind_av);
    SmallVector<AsyncValue*, 2> async_values;
    async_values.push_back(GetAsyncTensor());
    if (!IsDeviceAvailable()) {
      async_values.push_back(GetAsyncDevice().GetAsyncValue());
    }
    RunWhenReady(async_values, [th = CopyRef(),
                                result_ind_av = std::move(result_ind_av), dst,
                                exec_ctx]() {
      if (th.IsDeviceError()) {
        result_ind_av->ForwardTo(th.GetAsyncDevice().CopyRCRef());
        return;
      }
      if (th.GetAsyncTensor()->IsError()) {
        result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
        return;
      }
      auto& tensor = th.GetAsyncTensor()->get<Tensor>();
      if (dst.get() == th.GetAvailableDevice().get()) {
        result_ind_av->ForwardTo(FormRef(th.GetAsyncTensor()));
      } else {
        TensorType dst_tensor_type = InferDstTensorTypeFromDevice(*dst, tensor);
        result_ind_av->ForwardTo(ConvertTensor(
            exec_ctx, tensor, *th.GetAvailableDevice(), *dst, dst_tensor_type));
      }
    });
  }

  if (IsMetadataAvailable()) {
    return TensorHandle(std::move(dst), GetAvailableMetadata(),
                        std::move(result_tensor));
  } else {
    return TensorHandle(std::move(dst), GetAsyncMetadata().CopyRef(),
                        std::move(result_tensor));
  }
}

raw_ostream& operator<<(raw_ostream& os, const TensorHandle& handle) {
  auto tensor = handle.GetAsyncTensor();
  // Check for invalid states.  Both null could happen when in a moved-from
  // state.
  if (!handle.IsMetadataInline() && !handle.async_metadata_.GetAsyncValue() &&
      !tensor)
    return os << "NULL TensorHandle!";

  // Handle truly invalid states gracefully.
  if (!handle.IsMetadataInline() && !handle.async_metadata_.GetAsyncValue())
    return os << "Invalid TensorHandle with null metadata!";
  if (!tensor) return os << "Invalid TensorHandle with null tensor!";

  // If the tensor is resolved, just print it.
  if (handle.GetAsyncTensor()->IsConcrete()) return os << tensor->get<Tensor>();

  // If the tensor resolved to an error, print that.
  if (auto* error = tensor->GetErrorIfPresent())
    return os << "Error TensorHandle: '" << error->message << "'";

  // Otherwise, if the shape is present, print just that.  Note that there could
  // be a race between the check above and this check; we're ok with that.
  if (handle.IsMetadataInline())
    return os << "future TensorHandle with metadata "
              << handle.inlined_metadata_;
  else if (handle.async_metadata_.IsConcrete())
    return os << "future TensorHandle with metadata "
              << handle.async_metadata_.get();
  else if (auto* error = handle.async_metadata_.GetErrorIfPresent())
    return os << "future TensorHandle with error metadata '" << error->message
              << "'";

  return os << "fully future TensorHandle with unresolved metadata";
}

}  // namespace tfrt
