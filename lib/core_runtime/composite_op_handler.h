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

//===- composite_op_handler.h -----------------------------------*- C++ -*-===//
//
// This file declares CompositeOpHandler, responsible for executing a composite
// op.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_CORE_RUNTIME_COMPOSITE_OP_HANDLER_H_
#define TFRT_LIB_CORE_RUNTIME_COMPOSITE_OP_HANDLER_H_

#include <memory>

#include "llvm/ADT/StringMap.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/op_metadata_function.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class AsyncValue;
class Chain;
class Tensor;
class Function;
class HostContext;
class TensorHandle;

using DispatchFn = llvm::unique_function<void(
    ArrayRef<AsyncValue*>, const OpAttrsRef&, ArrayRef<TensorMetadata>,
    MutableArrayRef<RCReference<AsyncValue>>, AsyncValueRef<Chain>*, Location)>;

struct FunctionOpEntry {
  OpMetadataFn metadata_fn = nullptr;

  RCReference<Function> dispatch_fn;
};

class CompositeOpHandler : public OpHandler {
 public:
  static llvm::Expected<std::unique_ptr<CompositeOpHandler>> Create(
      CoreRuntime* runtime, OpHandler* fallback);

  explicit CompositeOpHandler(CoreRuntime* runtime);

  ~CompositeOpHandler() override;

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

  AsyncValueRef<HostTensor> CopyDeviceTensorToHost(
      const Tensor& tensor) override;

  AsyncValueRef<Tensor> CopyHostTensorToDevice(
      const DenseHostTensor& tensor) override;

  // TODO(fishx): Allow registering metadata function.

  bool RegisterCompositeOp(string_view name, RCReference<Function> fn);

 private:
  void ExecuteWithoutMetadata(
      Location loc, MutableArrayRef<TensorHandle> arguments,
      const OpAttrsRef& attrs, size_t num_results,
      SmallVectorImpl<AsyncValueRef<TensorMetadata>>* result_md_avs,
      SmallVectorImpl<AsyncValueRef<Tensor>>* result_tensor_avs,
      AsyncValueRef<Chain>* chain, DispatchFn dispatch_fn);

  llvm::StringMap<FunctionOpEntry> composite_op_mappings_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_CORE_RUNTIME_COMPOSITE_OP_HANDLER_H_
