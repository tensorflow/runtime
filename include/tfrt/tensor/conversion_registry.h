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
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {

class Device;
class DeviceType;
class ExecutionContext;

// TensorConversionFn allows convert a Tensor from a source Tensor type into a
// destination Tensor type. This can (in general) require
// device computation or lots of copying, so this returns an AsyncValue for the
// result.
//
// This returns an error value if the input tensor is invalid or an error is
// encountered like OOM.
// TODO(fishx): Change HostContext to ExecutionContext.
using TensorConversionFn = AsyncValueRef<Tensor> (*)(
    const Tensor& tensor, const Device& src, const Device& dst,
    TensorType dst_tensor_type, const ExecutionContext& exec_ctx);

// Convert tensor to tensor type. It will look up and call the
// TensorConversionFn registered in the TensorConversionFn registry
AsyncValueRef<Tensor> ConvertTensor(const ExecutionContext& exec_ctx,
                                    const Tensor& tensor, const Device& src,
                                    const Device& dst,
                                    TensorType dst_tensor_type);

// This variant is to support legacy cases that haven't been migrated to use
// ExecutionContext.
// TODO(fishx): Remove this method.
AsyncValueRef<Tensor> ConvertTensor(const Tensor& tensor, const Device& src,
                                    const Device& dst,
                                    TensorType dst_tensor_type,
                                    HostContext* host);

class TensorConversionFnRegistry {
 public:
  TensorConversionFnRegistry() = default;

  TensorConversionFnRegistry(const TensorConversionFnRegistry&) = delete;
  TensorConversionFnRegistry& operator=(const TensorConversionFnRegistry&) =
      delete;

  // The key of TensorConversionFnRegistry is a pair of source TensorType
  // destination TensorType.
  struct ConversionKey {
    TensorType src_tensor_type;
    TensorType dst_tensor_type;
  };

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
  static ConversionKey getEmptyKey() {
    return {tfrt::TensorType::kUnknownTensorType,
            tfrt::TensorType::kUnknownTensorType};
  }
  static ConversionKey getTombstoneKey() {
    return {tfrt::TensorType::kUnknownTensorType,
            tfrt::TensorType::kUnknownTensorType};
  }
  static unsigned getHashValue(const ConversionKey& k) {
    return detail::combineHashValue(
        static_cast<unsigned>(k.src_tensor_type.id()),
        static_cast<unsigned>(k.dst_tensor_type.id()));
  }
  static bool isEqual(const ConversionKey& lhs, const ConversionKey& rhs) {
    return lhs.src_tensor_type == rhs.src_tensor_type &&
           lhs.dst_tensor_type == rhs.dst_tensor_type;
  }
};
}  // namespace llvm

#endif  // TFRT_TENSOR_CONVERSION_REGISTRY_H_
