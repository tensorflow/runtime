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

//===- conversion_registry.h ------------------------------------*- C++ -*-===//
//
// This file defines Tensor Conversion Function and its registry.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_TENSOR_CONVERSION_REGISTRY_H_
#define TFRT_TENSOR_CONVERSION_REGISTRY_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {

class Device;
class DeviceType;
class ExecutionContext;

// It is a bitmask indicating supported tensor formats. It is a bitmask of
// tensor Subclass kinds.
struct TensorFormats {
  static TensorFormats Create(llvm::ArrayRef<Tensor::Subclass> sbuclasses);

  uint32_t allowed_formats;

  bool Contains(Tensor::Subclass subclass) const;
};

inline raw_ostream& operator<<(raw_ostream& os, const TensorFormats& formats) {
  os << llvm::format_hex(formats.allowed_formats, 2);
  return os;
}

// TensorConversionFn allows copying the contents of a Tensor into another
// Tenosr format and/or into another Device. This can (in general) require
// device computation or lots of copying, so this returns an AsyncValue for the
// result.
//
// This returns an error value if the input tensor is invalid or an error is
// encountered like OOM.
// TODO(fishx): Change HostContext to ExecutionContext.
using TensorConversionFn = AsyncValueRef<Tensor> (*)(
    const Tensor& tensor, const Device& src, const Device& dst,
    TensorFormats allowed_formats, const ExecutionContext& exec_ctx);

// Transfer tensor to device. It will look up and call the TensorConversionFn
// registered in the TensorConversionFn registry.
AsyncValueRef<Tensor> TransferTensorTo(const Tensor& tensor, const Device& src,
                                       const Device& dst,
                                       TensorFormats allowed_formats,
                                       HostContext* host);

class TensorConversionFnRegistry {
 public:
  TensorConversionFnRegistry() = default;

  TensorConversionFnRegistry(const TensorConversionFnRegistry&) = delete;
  TensorConversionFnRegistry& operator=(const TensorConversionFnRegistry&) =
      delete;

  // The key of TensorConversionFnRegistry is a pair of Tensor subclass of
  // source tensor and a target device type.
  struct ConversionKey {
    Tensor::Subclass src_subclass;
    const DeviceType* dst_device_type;
  };

  // TODO(fishx): Add a static check to ensure that the TensorConversionFn match
  // the TensorSubclass in its Key.
  void AddTensorConversionFn(ConversionKey key, TensorConversionFn fn);
  TensorConversionFn GetTensorConversionFn(ConversionKey key) const;

 private:
  llvm::DenseMap<ConversionKey, TensorConversionFn> conversion_fn_map_;
};

// The type for TensorConversionFn registration functions.
using TensorConversionFnRegistration = void (*)(TensorConversionFnRegistry*);

// This is called to register all the statically linked TensorConversionFn.
void RegisterTensorConversionFns(HostContext* host);

// Adds a TensorConversionFn to the registry.
void AddStaticTensorConversionFn(TensorConversionFnRegistration func);

}  // namespace tfrt

namespace llvm {

// This allows Tensor::Subclass to be used as keys in llvm::DenseMap.
template <>
struct DenseMapInfo<tfrt::TensorConversionFnRegistry::ConversionKey> {
  using ConversionKey = tfrt::TensorConversionFnRegistry::ConversionKey;
  using DeviceTypeInfo = DenseMapInfo<const tfrt::DeviceType*>;
  static ConversionKey getEmptyKey() {
    return {tfrt::Tensor::Subclass::DenseHost, DeviceTypeInfo::getEmptyKey()};
  }
  static ConversionKey getTombstoneKey() {
    return {tfrt::Tensor::Subclass::DenseHost, DeviceTypeInfo::getEmptyKey()};
  }
  static unsigned getHashValue(const ConversionKey& k) {
    return detail::combineHashValue(
        static_cast<unsigned>(k.src_subclass),
        DeviceTypeInfo::getHashValue(k.dst_device_type));
  }
  static bool isEqual(const ConversionKey& lhs, const ConversionKey& rhs) {
    return lhs.src_subclass == rhs.src_subclass &&
           DeviceTypeInfo::isEqual(lhs.dst_device_type, rhs.dst_device_type);
  }
};
}  // namespace llvm

#endif  // TFRT_TENSOR_CONVERSION_REGISTRY_H_
