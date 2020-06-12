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

//===- device_factory.h -----------------------------------------*- C++ -*-===//
//
// This file declares class OpHandlerFactory.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_CORE_RUNTIME_OP_HANDLER_FACTORY_H_
#define TFRT_CORE_RUNTIME_OP_HANDLER_FACTORY_H_

#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"

namespace tfrt {

class OpHandler;
class CoreRuntime;

using OpHandlerCreateFn =
    std::function<llvm::Expected<std::unique_ptr<OpHandler>>(CoreRuntime*,
                                                             OpHandler*)>;

// OpHandlerFactory maintains the factory methods for creating various types of
// op handlers.
class OpHandlerFactory {
 public:
  static OpHandlerFactory& GetGlobalOpHandlerFactory();

  void Add(string_view name, OpHandlerCreateFn create_fn) {
    mutex_lock lock(mu_);
    create_fns_.try_emplace(name, std::move(create_fn));
  }

  llvm::Expected<OpHandlerCreateFn> Get(string_view name) const {
    mutex_lock lock(mu_);
    auto iter = create_fns_.find(name);
    if (iter == create_fns_.end()) {
      return MakeStringError("Factory method for ", std::string(name).c_str(),
                             " not found. Available op handlers: ",
                             Join(create_fns_.keys(), ", "), ".\n");
    }
    return iter->second;
  }

 private:
  mutable mutex mu_;
  llvm::StringMap<OpHandlerCreateFn> create_fns_ TFRT_GUARDED_BY(mu_);
};

// A helper class for registering a OpHandler's factory method into the global
// factory.
struct OpHandlerRegistration {
  OpHandlerRegistration(string_view name, const OpHandlerCreateFn& create_fn) {
    OpHandlerFactory::GetGlobalOpHandlerFactory().Add(name, create_fn);
  }
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_HANDLER_FACTORY_H_
