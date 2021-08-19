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
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
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
  T& GetArgAt(int index) {
    return arguments_[index]->get<T>();
  }

  template <typename T>
  KernelRunner& AddAttribute(T value) {
    attr_offsets_.emplace_back(bef_attr_encoder_.EncodeAttr(value));
    return *this;
  }

  template <typename T>
  KernelRunner& AddArrayAttribute(llvm::ArrayRef<T> value);

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
  KernelRunner& AddRequestContextData(Args&&... args) {
    assert(!req_ctx_ &&
           "Cannot add request context data after req_ctx_ is materialized");
    req_ctx_builder_.context_data().emplace<T>(std::forward<Args>(args)...);
    return *this;
  }

  template <typename T, typename... Args>
  KernelRunner& AddResource(llvm::StringRef name, Args&&... args) {
    assert(!req_ctx_ && "Cannot add resource after req_ctx_ is materialized");
    resource_ctx_.CreateResource<T>(name, std::forward<Args>(args)...);
    return *this;
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

  BefAttrEncoder bef_attr_encoder_;
  std::vector<uint32_t> attr_offsets_;

  SmallVector<RCReference<AsyncValue>, 8> arguments_;
  SmallVector<RCReference<AsyncValue>, 8> results_;

  ResourceContext resource_ctx_;
  RequestContextBuilder req_ctx_builder_;
  RCReference<RequestContext> req_ctx_;
};

template <typename T>
KernelRunner& KernelRunner::AddArrayAttribute(llvm::ArrayRef<T> values) {
  attr_offsets_.emplace_back(bef_attr_encoder_.EncodeArrayAttr(values));
  return *this;
}

}  // namespace tfrt

#endif  // TFRT_UTILS_KERNEL_RUNNER_H_
