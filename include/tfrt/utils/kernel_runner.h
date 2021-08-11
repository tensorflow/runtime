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

#ifndef TFRT_UTILS_KERNEL_RUNNER_H_
#define TFRT_UTILS_KERNEL_RUNNER_H_

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace detail {

// An allocator that allocates data in a BefBuffer.
class BefBufferAllocator {
 public:
  template <typename T, typename... Args>
  T* construct(Args&&... args) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "BefBufferAllocator only supports trivially copyable types");
    return new (allocate<T>()) T(std::forward<Args>(args)...);
  }

  template <typename T>
  void* allocate() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "BefBufferAllocator only supports trivially copyable types");
    return allocate(sizeof(T), alignof(T));
  }

  void* allocate(size_t size, size_t align);

  ArrayRef<uint8_t> data() const {
    return {reinterpret_cast<const uint8_t*>(data_.data()), data_.size()};
  }
  ArrayRef<uint32_t> offsets() const { return offsets_; }

 private:
  BefBuffer data_;
  std::vector<uint32_t> offsets_;
};

}  // namespace detail

/**
 * KernelRunner allows user to run individual TFRT kernels in isolation in C++.
 * This is particularly useful for testing kernels that do not have a
 * human-readable MLIR format or testing kernels with large tensors.
 *
 * Example usage:
 *
 * ```
 * auto sum = KernelRunner("tfrt_test.sum")
 *               .SetArgs(1, 2, 3)
 *               .RunAndGetResult<int>();
 * EXPECT_EQ(sum, 6);
 *
 * auto str = KernelRunner("tfrt_test.get_string")
 *              .AddStringAttribute("hello")
 *              .RunAndGetResult<std::string>();
 *
 * EXPECT_EQ(str, "hello");
 * ```
 */

class KernelRunner {
 public:
  // If `host` is not provided, a default HostContext will be created.
  explicit KernelRunner(string_view name, HostContext* host = nullptr);

  template <typename... Args>
  KernelRunner& SetArgs(Args&&... args) {
    SetArgsHelper(std::forward<Args>(args)...);
    return *this;
  }

  template <typename T>
  KernelRunner& AddAttribute(T value) {
    allocator_.construct<T>(value);
    return *this;
  }

  KernelRunner& AddStringAttribute(string_view str);

  template <typename T>
  const T& RunAndGetResult() {
    Run(1);
    return GetResultAt<T>(0);
  }

  void Run(size_t num_results);

  template <typename T>
  const T& GetResultAt(int index) {
    return results_[index]->get<T>();
  }

  template <typename T, typename... Args>
  T& AddRequestContextData(Args&&... args) {
    return req_ctx_builder_.context_data().emplace<T>(
        std::forward<Args>(args)...);
  }

 private:
  template <typename T>
  void AddArg(T&& t) {
    arguments_.emplace_back(
        MakeAvailableAsyncValueRef<std::decay_t<T>>(std::forward<T>(t))
            .ReleaseRCRef());
  }

  void SetArgsHelper() {}

  template <typename T, typename... Args>
  void SetArgsHelper(T&& t, Args&&... args) {
    AddArg(std::forward<T>(t));
    SetArgsHelper(std::forward<Args>(args)...);
  }

  std::unique_ptr<HostContext> default_host_context_;
  HostContext* host_;
  AsyncKernelImplementation kernel_fn_;

  ResourceContext resource_ctx_;
  RequestContextBuilder req_ctx_builder_;
  detail::BefBufferAllocator allocator_;
  SmallVector<RCReference<AsyncValue>, 8> arguments_;
  SmallVector<RCReference<AsyncValue>, 8> results_;
};

}  // namespace tfrt

#endif  // TFRT_UTILS_KERNEL_RUNNER_H_
