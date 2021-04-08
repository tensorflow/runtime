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

// This file declares CoreRuntimeOp, responsible for executing ops.
#ifndef TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_
#define TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/host_context/device.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {

class TensorHandle;
class OpAttrsRef;
class Chain;
class ExecutionContext;

// A callable op handle prepared by a specific OpHandler.
class CoreRuntimeOp {
 public:
  // Default initialized CoreRuntimeOp's are in the invalid state.
  explicit CoreRuntimeOp();
  CoreRuntimeOp(llvm::unique_function<void(const OpInvocation&) const>&& fn,
                bool is_fallback);

  CoreRuntimeOp(llvm::unique_function<void(const OpInvocation&) const>&& fn,
                bool is_fallback, RCReference<Device> device,
                TensorType arg_tensor_type = DenseHostTensor::kTensorType);

  // Creates a "native function" in that it takes and returns AsyncValues of
  // any types, and not having to going through TensorHandle.
  explicit CoreRuntimeOp(
      llvm::unique_function<void(const CompositeOpInvocation&) const>&&
          native_fn);

  // Execute the prepared op.
  //
  // Executing an op is generally an asynchronous operation that will produce
  // TensorHandle's for the results of executing the operation (which can be
  // asynchronous errors), but op dispatch can also synchronously fail.  This
  // happens when the specified operation is not known, or when synchronously
  // executed logic (e.g. metadata functions) detect an error.  When this
  // happens, this method emits the corresponding error using the normal
  // diagnostic machinery and fills the `results` TensorHandle's and result
  // chain with the error value.
  //
  // Note that this takes the input argument TensorHandle's and is allowed to
  // destructively mutate them.  This is useful in the common case of expression
  // trees like (x+y)*z, but there are cases where the caller will need to
  // duplicate the TensorHandle (using CopyRef()) method if it needs the
  // TensorHandle to be valid after the execute call.
  void operator()(const ExecutionContext& exec_ctx,
                  MutableArrayRef<TensorHandle> arguments,
                  const OpAttrsRef& attrs,
                  MutableArrayRef<TensorHandle> results,
                  AsyncValueRef<Chain>* chain) const;

  void operator()(const OpInvocation& invocation) const;

  void operator()(const CompositeOpInvocation& invocation) const;

  explicit operator bool() const { return static_cast<bool>(fn_); }

  bool IsFallback() const { return is_fallback_; }

  string_view DeviceName() { return device_->name(); }

  RCReference<Device> GetDeviceRef() { return device_.CopyRef(); }

  const TensorType& GetTensorType() { return arg_tensor_type_; }

 private:
  // Since CoreRuntimeOp is semantically immutable, its operator() should be
  // const functions.  We need to mark fn_ and native_fn_ as mutable so we can
  // mark the operator() of this class as const function, because
  // llvm::unique_function::operator() is non-const for some reason.
  llvm::unique_function<void(const OpInvocation&) const> fn_;
  llvm::unique_function<void(const CompositeOpInvocation&) const> native_fn_;
  bool is_fallback_;

  // The target device that the op requires the arguments to be placed on.
  // Currently we assume all arguments are placed on the same device for one op.
  // this may not be true for certain ops (e.g. tf.Reshape). Currently such ops
  // are supported by TF Runtime Fallback. In the future, indivicual argument
  // placement information will be obtained from redesigned OpRegistry (SSOT).
  RCReference<Device> device_;

  // The target tensor type for arguments. By default DenseHostTensor on CPU.
  // Same as above, we currently assume all arguments have the same TensorType,
  // which may not be ture for certain ops, and will be obtained from redesigned
  // OpRegistry (SSOT).
  TensorType arg_tensor_type_ = DenseHostTensor::kTensorType;
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_CORE_RUNTIME_OP_H_
