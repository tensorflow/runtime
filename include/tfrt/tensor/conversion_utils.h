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

// This file defines macro to make it easier to write conversion function. This
// is intended to be small and simple things and is nearly header-only.
#ifndef TFRT_TENSOR_CONVERSION_UTILS_H_
#define TFRT_TENSOR_CONVERSION_UTILS_H_

#include <type_traits>

#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {

// TFRT_CONVERSION is a macro that makes defining conversion functions more
// straightforward. Example:
//
//   DenseGpuTensor DHTToDGTConversion(const DenseHostTensor& t,
//                                     const CpuDevice& src,
//                                     const GpuDevice& dst
//                                     const ExecutionContext& exec_ctx) { ... }
//
// registry->AddTensorConversionFn(TFRT_CONVERSION(DHTToDGTConversion));
#define TFRT_CONVERSION(...)                               \
  ::tfrt::ConversionFnImpl<decltype(&__VA_ARGS__),         \
                           &__VA_ARGS__>::ConversionKey(), \
      ::tfrt::ConversionFnImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke

template <typename F, F f>
struct ConversionFnImpl;

template <typename DeviceT,
          std::enable_if_t<!std::is_same<Device, DeviceT>::value, int> = 0>
void checkDeviceType(const Device& src) {
  assert(src.type() == DeviceT::kDeviceType);
}

template <typename DeviceT,
          std::enable_if_t<std::is_same<Device, DeviceT>::value, int> = 0>
void checkDeviceType(const Device& src) {}

template <typename ReturnT, typename TensorT, typename SrcDeviceT,
          typename DstDeviceT,
          ReturnT (*impl_fn)(const TensorT&, const SrcDeviceT&,
                             const DstDeviceT&, const ExecutionContext&)>
struct ConversionFnImpl<ReturnT (*)(const TensorT&, const SrcDeviceT&,
                                    const DstDeviceT&, const ExecutionContext&),
                        impl_fn> {
  static AsyncValueRef<Tensor> Invoke(const Tensor& tensor, const Device& src,
                                      const Device& dst,
                                      const ExecutionContext& exec_ctx) {
    static_assert(std::is_base_of<Tensor, TensorT>::value,
                  "the first argument must be subclass of Tensor");
    static_assert(std::is_base_of<Device, SrcDeviceT>::value,
                  "the second argument must be subclass of Device");
    static_assert(std::is_base_of<Device, DstDeviceT>::value,
                  "the third argument must be subclass of Device");
    assert(tensor.IsTensorType(TensorT::kTensorType));
    const auto& t = static_cast<const TensorT&>(tensor);
    checkDeviceType<SrcDeviceT>(src);
    const auto& s = static_cast<const SrcDeviceT&>(src);
    checkDeviceType<DstDeviceT>(dst);
    const auto& d = static_cast<const DstDeviceT&>(dst);
    return ReturnHelper<ReturnT>::handle(impl_fn(t, s, d, exec_ctx), exec_ctx);
  }

  static TensorConversionFnRegistry::ConversionKey ConversionKey() {
    return {TensorT::kTensorType, ReturnHelper<ReturnT>::type()};
  }

 private:
  template <typename ReturnTensorT>
  struct ReturnHelper {
    static_assert(std::is_base_of<Tensor, ReturnTensorT>::value,
                  "the result must be subclass of Tensor");
    static AsyncValueRef<Tensor> handle(ReturnTensorT v,
                                        const ExecutionContext& exec_ctx) {
      return MakeAvailableAsyncValueRef<ReturnTensorT>(
          exec_ctx.host(), std::forward<ReturnTensorT>(v));
    }

    static TensorType type() { return ReturnTensorT::kTensorType; }
  };

  template <typename ReturnTensorT>
  struct ReturnHelper<AsyncValueRef<ReturnTensorT>> {
    static_assert(std::is_base_of<Tensor, ReturnTensorT>::value,
                  "the result must be subclass of Tensor");
    static AsyncValueRef<Tensor> handle(AsyncValueRef<ReturnTensorT> v,
                                        const ExecutionContext& exec_ctx) {
      return v;
    }

    static TensorType type() { return ReturnTensorT::kTensorType; }
  };

  template <typename ReturnTensorT>
  struct ReturnHelper<llvm::Expected<ReturnTensorT>> {
    static AsyncValueRef<Tensor> handle(llvm::Expected<ReturnTensorT>&& v,
                                        const ExecutionContext& exec_ctx) {
      if (v) {
        return ReturnHelper<ReturnTensorT>::handle(std::move(*v), exec_ctx);
      } else {
        return EmitErrorAsync(exec_ctx, v.takeError());
      }
    }

    static TensorType type() { return ReturnTensorT::kTensorType; }
  };
};

}  // namespace tfrt
#endif  // TFRT_TENSOR_CONVERSION_UTILS_H_
