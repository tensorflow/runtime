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

//===- driver.h - CoreRuntime Driver in C++ ---------------------*- C++ -*-===//
//
// This file defines a simple Driver that allows executing op-by-op using
// CoreRuntime API in C++.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_EXAMPLES_CORE_RUNTIME_DRIVER_H_
#define TFRT_EXAMPLES_CORE_RUNTIME_DRIVER_H_

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

class CoreRuntimeOp;
class OpHandler;
class OpAttrsRef;
class TensorHandle;

namespace example {

class CoreRuntimeDriver final : public LocationHandler {
 public:
  explicit CoreRuntimeDriver(const std::string& op_handlers);

  // Eagerly execute the op.
  void Execute(string_view op_name, Location loc,
               MutableArrayRef<TensorHandle> args, const OpAttrsRef& attrs,
               MutableArrayRef<TensorHandle> results);

  CoreRuntimeOp MakeOp(string_view op_name);
  HostContext* GetHostContext() const { return corert_->GetHostContext(); }

  // Create a encoded Location object.
  Location CreateLocation(const char* filename, int line_number);

  void WaitForHostContextQuiesce();

 private:
  CoreRuntimeDriver(std::unique_ptr<CoreRuntime> corert,
                    const std::string& op_handler);

  // Decode the location sections to figure out the file/line/column of this
  // error.
  DecodedLocation DecodeLocation(Location loc) const override;

  std::unique_ptr<CoreRuntime> corert_;
  OpHandler* op_handler_;
  AsyncValueRef<Chain> chain_;

  // TODO(b/147629198): Clean up this vector.
  std::vector<std::pair<const char*, int>> locations_;
};
}  // namespace example
}  // namespace tfrt

#endif  // TFRT_EXAMPLES_CORE_RUNTIME_DRIVER_H_
