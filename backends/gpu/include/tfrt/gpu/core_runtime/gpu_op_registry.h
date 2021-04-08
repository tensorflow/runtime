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

// This file declares GpuOpRegistry, which maps an op to an optional metadata
// function (the "shape" function that also produces layout and dtype) and a
// kernel dispatch function.

#ifndef TFRT_GPU_CORE_RUNTIME_GPU_OP_REGISTRY_H_
#define TFRT_GPU_CORE_RUNTIME_GPU_OP_REGISTRY_H_

#include "tfrt/core_runtime/op_metadata_function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
class AsyncValue;
class Chain;
class CoreRuntime;
class Tensor;
class ExecutionContext;
class OpAttrsRef;
class OpHandler;
class TensorHandle;
class TensorMetadata;
class GpuDispatchContext;

using GpuDispatchFn = void (*)(const ExecutionContext& exec_ctx,
                               GpuDispatchContext* dctx,
                               ArrayRef<AsyncValue*> inputs,
                               const OpAttrsRef& attrs,
                               ArrayRef<TensorMetadata> result_mds,
                               MutableArrayRef<RCReference<AsyncValue>> results,
                               AsyncValueRef<Chain>* chain);

// This represents a mapping from op names to the associated metadata functions
// (optional) and kernel dispatch functions.
class GpuOpRegistry {
 public:
  GpuOpRegistry();
  ~GpuOpRegistry();
  GpuOpRegistry(GpuOpRegistry&& other);
  GpuOpRegistry& operator=(GpuOpRegistry&& other);

  // Add an op with the specified dispatch function.
  void AddOp(string_view op_name, GpuDispatchFn dispatch_fn);

  void AddOp(string_view op_name, GpuDispatchFn dispatch_fn,
             ArrayRef<string_view> attr_names);

  // Set a metadata function for the specified op_name.  All metadata functions
  // are required to be semantically equal, so multiple registrations for the
  // same op are allowed (making static initialization easier).
  void AddMetadataFn(string_view op_name, OpMetadataFn metadata_fn);

 private:
  friend class GpuOpHandler;
  GpuOpRegistry(const GpuOpRegistry&) = delete;
  GpuOpRegistry& operator=(const GpuOpRegistry&) = delete;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Use this macro to add a function that will register ops that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the tfrt::OpRegistration alias.
#define TFRT_STATIC_GPU_OP_REGISTRATION(FUNC) \
  TFRT_STATIC_GPU_OP_REGISTRATION_(FUNC, __COUNTER__)
#define TFRT_STATIC_GPU_OP_REGISTRATION_(FUNC, N) \
  TFRT_STATIC_GPU_OP_REGISTRATION__(FUNC, N)
#define TFRT_STATIC_GPU_OP_REGISTRATION__(FUNC, N)      \
  static bool tfrt_static_op_##N##_registered_ = []() { \
    ::tfrt::AddStaticGpuOpRegistration(FUNC);           \
    return true;                                        \
  }()

// The type for op registration functions. This is the same as the
// prototype for the entry point function for dynamic plugins.
using GpuOpRegistration = void (*)(GpuOpRegistry*);

// Adds an op to the registry. This should not be used directly; use
// TFRT_STATIC_GPU_OP_REGISTRATION instead.
void AddStaticGpuOpRegistration(GpuOpRegistration func);

// This is called to register all the statically linked ops in the given
// registry.
void RegisterStaticGpuOps(GpuOpRegistry* op_reg);

// Create a OpHandler from the specified registry of GPU ops and kernels.  The
// op_handler is owned by CoreRuntime, just like all op_handlers.
tfrt::Expected<OpHandler*> CreateGpuOpHandler(GpuOpRegistry&& op_registry,
                                              CoreRuntime* runtime);

}  // namespace tfrt

#endif  // TFRT_GPU_CORE_RUNTIME_GPU_OP_REGISTRY_H_
