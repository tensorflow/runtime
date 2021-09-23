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

// This file declares CreateCpuOpHandler, which creates CPU Op Handler
// responsible for executing ops on CPU.

#ifndef TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_HANDLER_H_
#define TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_HANDLER_H_

#include <memory>

#include "llvm/ADT/SetVector.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"

namespace tfrt {
class CoreRuntime;
class CoreRuntimeOp;
class Device;

class CpuOpHandler : public OpHandler {
 public:
  static const char* const kName;
  ~CpuOpHandler() override {}

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  RCReference<Device> GetDeviceRef() { return device_; }

  void AddImplicitConversion(TensorType src, TensorType dst);

  bool AllowImplicitConversion(TensorType src, TensorType dst);

 private:
  const CpuOpRegistry op_registry_;
  RCReference<Device> device_;

  llvm::SmallSetVector<TensorConversionFnRegistry::ConversionKey, 4>
      allowed_conversions;

  friend llvm::Expected<CpuOpHandler*> CreateCpuOpHandler(
      CoreRuntime* runtime, RCReference<Device> device, OpHandler* fallback);

  explicit CpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                        CpuOpRegistry op_registry, RCReference<Device> device);
};

llvm::Expected<CpuOpHandler*> CreateCpuOpHandler(CoreRuntime* runtime,
                                                 RCReference<Device> device,
                                                 OpHandler* fallback);

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_HANDLER_H_
