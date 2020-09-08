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

//===- resource_context.h ---------------------------------------*- C++ -*-===//
//
// This file declares ResourceContext - a type-erased container for storing and
// retrieving resources.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_RESOURCE_CONTEXT_H_
#define TFRT_HOST_CONTEXT_RESOURCE_CONTEXT_H_

#include <array>
#include <atomic>
#include <memory>

#include "llvm/ADT/StringMap.h"
#include "llvm_derived/Support/unique_any.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

// ResourceContext is used to store and retrieve resources.
class ResourceContext {
 public:
  // Get a resource T with a `resource_name`. Thread-safe.
  template <typename T>
  Optional<T*> GetResource(string_view resource_name) TFRT_EXCLUDES(mu_) {
    tfrt::mutex_lock lock(mu_);
    auto it = resources_.find(resource_name);
    if (it == resources_.end()) {
      return llvm::None;
    }
    T* data = tfrt::any_cast<T>(&it->second);
    return data;
  }

  // Get a resource T with a `resource_name`. Asserts that the resource has
  // been created.
  // Thread-safe.
  template <typename T>
  T* GetResourceOrDie(tfrt::string_view resource_name) TFRT_EXCLUDES(mu_) {
    tfrt::mutex_lock lock(mu_);
    auto it = resources_.find(resource_name);
    assert(it != resources_.end());
    T* data = tfrt::any_cast<T>(&it->second);
    return data;
  }

  // Create a resource T with a `resource_name`.
  // Thread-safe.
  template <typename T, typename... Args>
  T* CreateResource(tfrt::string_view resource_name, Args&&... args)
      TFRT_EXCLUDES(mu_) {
    tfrt::mutex_lock lock(mu_);
    auto res = resources_.try_emplace(resource_name, tfrt::in_place_type<T>,
                                      std::forward<Args>(args)...);
    assert(res.second);
    return tfrt::any_cast<T>(&res.first->second);
  }

  // Get or create a resource T with a `resource_name`.
  // GetResource and CreateResource are the preferred API. GetOrCreateResource
  // is useful when callers want to lazily initialize some resources. Since it
  // requires constructor arguments, it is more awkward to use.
  // Thread-safe.
  template <typename T, typename... Args>
  T* GetOrCreateResource(tfrt::string_view resource_name, Args&&... args)
      TFRT_EXCLUDES(mu_) {
    tfrt::mutex_lock lock(mu_);
    auto res = resources_.try_emplace(resource_name, tfrt::in_place_type<T>,
                                      std::forward<Args>(args)...);
    return tfrt::any_cast<T>(&res.first->second);
  }

 private:
  tfrt::mutex mu_;
  llvm::StringMap<tfrt::UniqueAny> resources_ TFRT_GUARDED_BY(mu_);
};

}  // namespace tfrt
#endif  // TFRT_HOST_CONTEXT_RESOURCE_CONTEXT_H_
