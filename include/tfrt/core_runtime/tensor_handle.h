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

//===- tensor_handle.h ------------------------------------------*- C++ -*-===//
//
// This file declares the TensorHandle interface.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_CORE_RUNTIME_TENSOR_HANDLE_H_
#define TFRT_CORE_RUNTIME_TENSOR_HANDLE_H_

#include <memory>

#include "llvm/ADT/PointerIntPair.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/device.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {

class Device;
class Tensor;

// An opaque representation of a rectangular tensor computed by the host/device
// runtime.
class TensorHandle final {
 public:
  // Default initialized TensorHandle's are in the invalid state.
  explicit TensorHandle() : tensor_and_is_metadata_inline_(nullptr, false) {
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(RCReference<AsyncValue>());
  }

  // A TensorHandle owns a `async_metadata`, `tensor` and 'device', none of
  // these input pointer is allowed to be NULL.
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

  bool IsMetadataAvailable() const {
    return IsMetadataInline() || async_metadata_.IsConcrete();
  }

  bool IsMetadataError() const {
    return !IsMetadataInline() && async_metadata_.IsError();
  }

  const AsyncValueRef<TensorMetadata>& GetAsyncMetadata() const;

  // Return reference of the TensorMetadata.
  // Use this method only if IsMetadataAvailable() returns true.
  const TensorMetadata& GetAvailableMetadata() const {
    assert(IsValid());
    if (IsMetadataInline()) return inlined_metadata_;
    return async_metadata_.get();
  }

  // Returns nullptr if handle is in an invalid state.
  AsyncValue* GetAsyncTensor() const {
    return tensor_and_is_metadata_inline_.getPointer();
  }

  // Return a new copy of the TensorHandle without deep copying any of the
  // underlying data.
  TensorHandle CopyRef() const;

  // Return the tensor async value and drop this TensorHandle into its null
  // state.
  AsyncValueRef<Tensor> ReleaseTensorRef();

  // Only valid TensorHandles can be constructed, but a handle can become
  // invalid after it's moved.
  bool IsValid() const { return GetAsyncTensor() != nullptr; }

  const Device& device() const { return *device_; }

  // Transfer the TensorHandle to the target Device and convert its format. The
  // target device can be same as current device, in this case, it will only
  // do format conversion. If both target device and target format are same as
  // this TensorHandle, it will simply return a new reference of this
  // TensorHandle. If it fails, the result TensorHandle will contain the error
  // in its tensor_ field.
  TensorHandle TransferTo(const ExecutionContext& exec_ctx,
                          RCReference<Device> dst,
                          TensorType dst_tensor_type) const;

 private:
  friend raw_ostream& operator<<(raw_ostream& os, const TensorHandle& handle);

  bool IsMetadataInline() const {
    return tensor_and_is_metadata_inline_.getInt();
  }

  // Reset both tensor and metadata to default initialized state.
  void Reset() {
    if (IsMetadataInline()) {
      inlined_metadata_.~TensorMetadata();
    } else {
      async_metadata_.~AsyncValueRef();
    }
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(RCReference<AsyncValue>());
    tensor_and_is_metadata_inline_.setPointerAndInt(nullptr, false);
  }

  // This is a PointerIntPair of an AsyncValue* to the Tensor object and a bool
  // flag on whether metadata is inline. The tensor AsyncValue is null if the
  // handle is invalid.
  llvm::PointerIntPair<AsyncValue*, 1, bool> tensor_and_is_metadata_inline_;

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

  // The device where the underlying tensor is located on.
  RCReference<Device> device_;
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

  // DropRef on Tensor AsyncValue.
  auto tensor = GetAsyncTensor();
  if (tensor) tensor->DropRef();
}

inline TensorHandle::TensorHandle(TensorHandle&& other)
    : tensor_and_is_metadata_inline_(other.tensor_and_is_metadata_inline_),
      device_(std::move(other.device_)) {
  if (other.IsMetadataInline()) {
    new (&inlined_metadata_) TensorMetadata(std::move(other.inlined_metadata_));
  } else {
    new (&async_metadata_)
        AsyncValueRef<TensorMetadata>(std::move(other.async_metadata_));
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

  auto tensor = GetAsyncTensor();
  if (tensor) tensor->DropRef();
  tensor_and_is_metadata_inline_ = other.tensor_and_is_metadata_inline_;
  device_ = std::move(other.device_);
  // Reset other to default initialized state.
  other.Reset();
  return *this;
}

inline const AsyncValueRef<TensorMetadata>& TensorHandle::GetAsyncMetadata()
    const {
  assert(IsValid() && !IsMetadataInline());

  return async_metadata_;
}

inline TensorHandle TensorHandle::CopyRef() const {
  auto tensor = AsyncValueRef<Tensor>(FormRef(GetAsyncTensor()));
  if (IsMetadataInline())
    return TensorHandle(device_.CopyRef(), inlined_metadata_,
                        std::move(tensor));
  return TensorHandle(device_.CopyRef(), async_metadata_.CopyRef(),
                      std::move(tensor));
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
