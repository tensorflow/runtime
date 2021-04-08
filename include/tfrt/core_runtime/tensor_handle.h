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

// This file declares the TensorHandle interface.

#ifndef TFRT_CORE_RUNTIME_TENSOR_HANDLE_H_
#define TFRT_CORE_RUNTIME_TENSOR_HANDLE_H_

#include <memory>

#include "llvm/ADT/PointerIntPair.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/device.h"
#include "tfrt/tensor/tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {

class Device;

// An opaque representation of a rectangular tensor computed by the host/device
// runtime.
class TensorHandle final {
 public:
  // Default initialized TensorHandle's are in the invalid state.
  explicit TensorHandle() : tensor_and_flags_(nullptr, 0) {
    new (&async_metadata_) AsyncValueRef<TensorMetadata>();
    new (&async_device_) AsyncValueRef<RCReference<Device>>();
  }

  // A TensorHandle owns a `async_metadata`, `tensor` and 'device', none of
  // these input pointer is allowed to be NULL.
  TensorHandle(AsyncValueRef<RCReference<Device>> async_device,
               AsyncValueRef<TensorMetadata> async_metadata,
               AsyncValueRef<Tensor> tensor);

  TensorHandle(AsyncValueRef<RCReference<Device>> async_device,
               const TensorMetadata& metadata, AsyncValueRef<Tensor> tensor);

  TensorHandle(RCReference<Device> device,
               AsyncValueRef<TensorMetadata> async_metadata,
               AsyncValueRef<Tensor> tensor);

  TensorHandle(RCReference<Device> device, const TensorMetadata& metadata,
               AsyncValueRef<Tensor> tensor);

  // TODO(fishx): Change the argument to RCReference<ErrorAsyncValue> since
  // right now we cannot convert an AsyncValueRef to
  // RCReference<ErrorAsyncValue> easily.
  explicit TensorHandle(AsyncValueRef<TensorHandle> error);

  ~TensorHandle();

  // Move operations are supported.
  TensorHandle(TensorHandle&& other);

  TensorHandle& operator=(TensorHandle&& other);

  // This class is not copyable or assignable, but you can use CopyRef to
  // explicitly copy it.
  TensorHandle(const TensorHandle& other) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  // Create a TensorHandle from an error AsyncValue. Both metadata and tensor
  // will be set to the error AsyncValue.
  static TensorHandle CreateError(RCReference<AsyncValue> error);

  bool IsDeviceAvailable() const {
    return IsDeviceInline() || async_device_.IsConcrete();
  }

  bool IsDeviceError() const {
    return !IsDeviceInline() && async_device_.IsError();
  }

  const AsyncValueRef<RCReference<Device>>& GetAsyncDevice() const;

  // Return reference of the Device.
  // Use this method only if IsDeviceAvailable() returns true.
  const RCReference<Device>& GetAvailableDevice() const {
    assert(IsValid());
    if (IsDeviceInline()) return inlined_device_;
    return async_device_.get();
  }

  bool IsMetadataAvailable() const {
    // TODO(tfrt-devs): In the future improvement of TensorHandle semantics, we
    // should remove the async_metadata_ field. Because we should always get the
    // metadata from the tensor unless the metadata is already available, in
    // which case the metadata should be provided to the c'tor at the
    // construction time.
    return IsMetadataInline() || async_metadata_.IsConcrete() ||
           GetAsyncTensor()->IsConcrete();
  }

  bool IsMetadataError() const {
    return !IsMetadataInline() && async_metadata_.IsError() &&
           GetAsyncTensor()->IsError();
  }

  const AsyncValueRef<TensorMetadata>& GetAsyncMetadata() const;

  // Return reference of the TensorMetadata.
  // Use this method only if IsMetadataAvailable() returns true.
  const TensorMetadata& GetAvailableMetadata() const {
    assert(IsValid());
    if (IsMetadataInline()) return inlined_metadata_;
    if (async_metadata_.IsConcrete()) return async_metadata_.get();
    return GetAsyncTensor()->get<Tensor>().metadata();
  }

  // Returns nullptr if handle is in an invalid state.
  AsyncValue* GetAsyncTensor() const { return tensor_and_flags_.getPointer(); }

  // Returns true if this is an error TensorHandle
  bool IsError() const {
    return IsMetadataError() || GetAsyncTensor()->IsError();
  }

  // Returns the error in this TensorHandle
  ErrorAsyncValue* GetErrorAsyncValue();

  // Return a new copy of the TensorHandle without deep copying any of the
  // underlying data.
  TensorHandle CopyRef() const;

  // Return the tensor async value and drop this TensorHandle into its null
  // state.
  AsyncValueRef<Tensor> ReleaseTensorRef();

  // Only valid TensorHandles can be constructed, but a handle can become
  // invalid after it's moved.
  bool IsValid() const { return GetAsyncTensor() != nullptr; }

  // Transfer the TensorHandle to the target Device and convert its format. The
  // target device can be same as current device, in this case, it will only
  // do format conversion. If both target device and target format are same as
  // this TensorHandle, it will simply return a new reference of this
  // TensorHandle. If it fails, the result TensorHandle will contain the error
  // in its tensor_ field.
  TensorHandle TransferTo(const ExecutionContext& exec_ctx,
                          RCReference<Device> dst,
                          TensorType dst_tensor_type) const;

  TensorHandle TransferToSameDevice(const ExecutionContext& exec_ctx,
                                    TensorType dst_tensor_type) const;

  // Destination tensor type is inferred from device.
  TensorHandle TransferToInferredType(const ExecutionContext& exec_ctx,
                                      RCReference<Device> dst) const;

 private:
  friend raw_ostream& operator<<(raw_ostream& os, const TensorHandle& handle);

  bool IsDeviceInline() const {
    return tensor_and_flags_.getInt() & Flags::DeviceInline;
  }

  bool IsMetadataInline() const {
    return tensor_and_flags_.getInt() & Flags::MetadataInline;
  }

  // Reset both tensor and metadata to default initialized state.
  void Reset() {
    if (IsMetadataInline()) {
      inlined_metadata_.~TensorMetadata();
    } else {
      async_metadata_.~AsyncValueRef();
    }
    if (IsDeviceInline()) {
      inlined_device_.~RCReference();
    } else {
      async_device_.~AsyncValueRef();
    }
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(RCReference<AsyncValue>());
    new (&async_device_)
        AsyncValueRef<RCReference<Device>>(RCReference<AsyncValue>());
    tensor_and_flags_.setPointerAndInt(nullptr, 0);
  }

  enum Flags : uint32_t {
    MetadataInline = 1 << 0,
    DeviceInline = 1 << 1,
  };

  // This is a PointerIntPair of an AsyncValue* to the Tensor object and a two
  // bits flag to indicate whether metadata and device are inlined. The tensor
  // AsyncValue is null if the handle is invalid.
  llvm::PointerIntPair<AsyncValue*, 2, uint32_t> tensor_and_flags_;

  // If IsMetadataInline() is true, `inlined_metadata_` specifies the shape
  // and the dtype for the TensorHandle inline. Otherwise, `async_metadata_`
  // specifies the metadata in an AsyncValue, which may be null for an invalid
  // TensorHandle.
  //
  // Eager op dispatch aims to make sure the metadata for a TensorHandle is
  // always synchronously available, but certain ops (e.g. those with
  // data-dependent shapes) prevent this. As such, all clients should be
  // prepared to handle a 'future' metadata.
  //
  // `async_metadata_` is null and IsMetadataInline() is false for an invalid
  // TensorHandle.
  union {
    AsyncValueRef<TensorMetadata> async_metadata_;
    TensorMetadata inlined_metadata_;
  };

  // If IsDeviceInline() is true, `inlined_device_` specifies the device
  // for the TensorHandle inline. Otherwise, `async_device_`  specifies the
  // device in an AsyncValue, which may be null for an invalid TensorHandle.
  //
  // Eager op dispatch aims to make sure the device for a TensorHandle is
  // always synchronously available, but certain cases (e.g. asychronous graph
  // lowering) prevent this. As such, all clients should be prepared to handle
  // a 'future' metadata.
  //
  // `async_device_` is null and IsDeviceInline() is false for an invalid
  // TensorHandle.
  union {
    AsyncValueRef<RCReference<Device>> async_device_;
    RCReference<Device> inlined_device_;
  };
};

static_assert(sizeof(TensorHandle) == 40 || sizeof(void*) != 8,
              "Unexpected size for TensorHandle. TensorHandle should be 40 "
              "bytes on 64-bit architecture.");

inline TensorHandle::~TensorHandle() {
  if (IsMetadataInline()) {
    inlined_metadata_.~TensorMetadata();
  } else {
    async_metadata_.~AsyncValueRef();
  }
  if (IsDeviceInline()) {
    inlined_device_.~RCReference();
  } else {
    async_device_.~AsyncValueRef();
  }

  // DropRef on Tensor AsyncValue.
  auto tensor = GetAsyncTensor();
  if (tensor) tensor->DropRef();
}

inline TensorHandle::TensorHandle(TensorHandle&& other)
    : tensor_and_flags_(other.tensor_and_flags_) {
  if (other.IsMetadataInline()) {
    new (&inlined_metadata_) TensorMetadata(std::move(other.inlined_metadata_));
  } else {
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(std::move(other.async_metadata_));
  }
  if (other.IsDeviceInline()) {
    new (&inlined_device_)
        RCReference<Device>(std::move(other.inlined_device_));
  } else {
    new (&async_device_)
        AsyncValueRef<RCReference<Device>>(std::move(other.async_device_));
  }
  // Reset other to default initialized state.
  other.Reset();
}

inline TensorHandle& TensorHandle::operator=(TensorHandle&& other) {
  if (IsMetadataInline() && other.IsMetadataInline()) {
    inlined_metadata_ = std::move(other.inlined_metadata_);
  } else if (!IsMetadataInline() && !other.IsMetadataInline()) {
    async_metadata_ = std::move(other.async_metadata_);
  } else if (IsMetadataInline() && !other.IsMetadataInline()) {
    inlined_metadata_.~TensorMetadata();
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(std::move(other.async_metadata_));
  } else {  // !IsMetadataInline() && other.IsMetadataInline()
    async_metadata_.~AsyncValueRef();
    new (&inlined_metadata_) TensorMetadata(std::move(other.inlined_metadata_));
  }

  if (IsDeviceInline() && other.IsDeviceInline()) {
    inlined_device_ = std::move(other.inlined_device_);
  } else if (!IsDeviceInline() && !other.IsDeviceInline()) {
    async_device_ = std::move(other.async_device_);
  } else if (IsDeviceInline() && !other.IsDeviceInline()) {
    inlined_device_.~RCReference();
    new (&async_device_)
        AsyncValueRef<RCReference<Device>>(std::move(other.async_device_));
  } else {  // !IsDeviceInline() && other.IsDeviceInline()
    async_device_.~AsyncValueRef();
    new (&inlined_device_)
        RCReference<Device>(std::move(other.inlined_device_));
  }

  auto tensor = GetAsyncTensor();
  if (tensor) tensor->DropRef();
  tensor_and_flags_ = other.tensor_and_flags_;
  // Reset other to default initialized state.
  other.Reset();
  return *this;
}

inline const AsyncValueRef<RCReference<Device>>& TensorHandle::GetAsyncDevice()
    const {
  assert(IsValid() && !IsDeviceInline());

  return async_device_;
}

inline const AsyncValueRef<TensorMetadata>& TensorHandle::GetAsyncMetadata()
    const {
  assert(IsValid() && !IsMetadataInline());

  return async_metadata_;
}

inline TensorHandle TensorHandle::CopyRef() const {
  auto tensor = AsyncValueRef<Tensor>(FormRef(GetAsyncTensor()));
  if (IsDeviceInline() && IsMetadataInline()) {
    return TensorHandle(inlined_device_.CopyRef(), inlined_metadata_,
                        std::move(tensor));
  } else if (!IsDeviceInline() && IsMetadataInline()) {
    return TensorHandle(async_device_.CopyRef(), inlined_metadata_,
                        std::move(tensor));
  } else if (IsDeviceInline() && !IsMetadataInline()) {
    return TensorHandle(inlined_device_.CopyRef(), async_metadata_.CopyRef(),
                        std::move(tensor));
  } else {
    return TensorHandle(async_device_.CopyRef(), async_metadata_.CopyRef(),
                        std::move(tensor));
  }
}

// Release the tensor and put the handle in a default-constructed state.
inline AsyncValueRef<Tensor> TensorHandle::ReleaseTensorRef() {
  auto tensor = AsyncValueRef<Tensor>(TakeRef(GetAsyncTensor()));
  // Reset to a default-constructed state.
  Reset();
  return tensor;
}

raw_ostream& operator<<(raw_ostream& os, const TensorHandle& handle);

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_TENSOR_HANDLE_H_
