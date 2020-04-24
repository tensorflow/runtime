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

//===- cpu_op_handler.h -----------------------------------------*- C++ -*-===//
//
// This file declares CpuOpHandler, responsible for executing ops on CPU.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_HANDLER_H_
#define TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_HANDLER_H_

#include <memory>

#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"

namespace tfrt {

class AsyncValue;
class Chain;
class CpuOpRegistry;
class Tensor;

class CpuOpHandler : public OpHandler {
 public:
  static llvm::Expected<std::unique_ptr<CpuOpHandler>> Create(
      CoreRuntime* runtime, OpHandler* fallback);

  explicit CpuOpHandler(CoreRuntime* runtime, OpHandler* fallback,
                        CpuOpRegistry op_registry);

  ~CpuOpHandler() override;

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  // For CpuOpHandler, the argument `tensor` needs to be a HostTensor. This
  // function returns a DenseHostTensor that contains a copy of the underlying
  // buffer.
  AsyncValueRef<DenseHostTensor> CopyDeviceTensorToHost(
      const Tensor& tensor) override;

  // This function returns a DenseHostTensor that contains a copy of the
  // underlying buffer of the argument `tensor`.
  AsyncValueRef<Tensor> CopyHostTensorToDevice(
      const DenseHostTensor& tensor) override;

 private:
  const CpuOpRegistry op_registry_;
};

}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_CORE_RUNTIME_CPU_OP_HANDLER_H_
