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

//===- cpu_op_registry.h ----------------------------------------*- C++ -*-===//
//
// This file declares CpuOpRegistry, which maps an op to an optional metadata
// function (the "shape" function that also produces layout and dtype) and a
// kernel dispatch function.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_REGISTRY_H_
#define TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_REGISTRY_H_

#include <memory>

#include "tfrt/core_runtime/op_metadata_function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class AsyncValue;
class Chain;
class CoreRuntime;
class HostContext;
class HostTensor;
class ExecutionContext;
class OpAttrsRef;
class OpHandler;
class TensorHandle;
class TensorMetadata;

// This is the signature implemented by all CPU ops.  They take Tensor buffers
// inputs and allocate and return tensors for their results.  If the op has a
// metadata function, then the result of the function is passed in as
// result_mds, otherwise it is an empty list.
//
// If the kernel has a runtime failure, the chain should be set to the
// error value, and any invalid results should be set to errors as well.
// TODO(b/153484730): Remove duplicate HostContext. Today we need this to make
// CpuDispatchFn has the same signature as GpuDispatchFn.
//
// TODO(b/154970304): Move ExecutionContext to be the first argument to keep all
// input arguments before output arguments. We also want to establish a
// convention of placing ExecutionContext as the first argument.
using CpuDispatchFn = void (*)(const ExecutionContext& exec_ctx,
                               ArrayRef<AsyncValue*> inputs,
                               const OpAttrsRef& attrs,
                               ArrayRef<TensorMetadata> result_mds,
                               MutableArrayRef<RCReference<AsyncValue>> results,
                               AsyncValueRef<Chain>* chain);

// CpuOpFlags allows customization points for ops that want to support
// more exotic features.  The defaults are set to conservatively correct
// behavior:
//
//   1) we assume there are side-effects unless told otherwise.
//   2) we assume op implementations only work with DenseHostTensor.
//
struct CpuOpFlags {
  enum Flags : uint32_t {
    None = 0,

    // If this is set, the op implementation declares that it is side-effect
    // free.  This means it will not get passed in a chain, and does not get
    // sequence w.r.t. other side effecting ops.
    NoSideEffects = 1 << 0,

    // If this is set, the op dispatch function is prepared to deal with
    // tensor inputs in ScalarHostTensor format.
    AllowsScalar = 1 << 1,

    // If this is set, the op dispatch function is prepared to deal with
    // tensor inputs in StringHostTensor format.
    AllowsString = 1 << 2,

    // If this is set, the op dispatch function is prepared to deal with
    // tensor inputs in CooHostTensor format.
    AllowsCoo = 1 << 3,

    // If this is set, the op dispatch function is prepared to deal with
    // tensor inputs in TfLiteHostTensor format.
    AllowsTfLite = 1 << 4,
  } flags;

  explicit CpuOpFlags() : flags(None) {}
  /*implicit*/ CpuOpFlags(Flags flags) : flags(flags) {}

  explicit operator bool() const { return flags != 0; }

  CpuOpFlags operator&(CpuOpFlags rhs) const {
    auto result = static_cast<int>(flags) & static_cast<int>(rhs.flags);
    return CpuOpFlags(static_cast<Flags>(result));
  }
  CpuOpFlags operator|(CpuOpFlags rhs) const {
    auto result = static_cast<int>(flags) | static_cast<int>(rhs.flags);
    return CpuOpFlags(static_cast<Flags>(result));
  }
};

inline CpuOpFlags operator&(CpuOpFlags::Flags lhs, CpuOpFlags::Flags rhs) {
  return CpuOpFlags(lhs) & CpuOpFlags(rhs);
}

inline CpuOpFlags operator|(CpuOpFlags::Flags lhs, CpuOpFlags::Flags rhs) {
  return CpuOpFlags(lhs) | CpuOpFlags(rhs);
}

// This represents a mapping from op names to the associated metadata functions
// (optional) and kernel dispatch functions.
class CpuOpRegistry {
 public:
  CpuOpRegistry();
  ~CpuOpRegistry();
  CpuOpRegistry(CpuOpRegistry&& other);
  CpuOpRegistry& operator=(CpuOpRegistry&& other);

  // Add an op with the specified dispatch function.  This style of dispatch
  // function does not require a metadata function.
  void AddOp(string_view op_name, CpuDispatchFn dispatch_fn, CpuOpFlags flags);

  void AddOp(string_view op_name, CpuDispatchFn dispatch_fn, CpuOpFlags flags,
             ArrayRef<string_view> attr_names);

  // Set a metadata function for the specified op_name.  All metadata functions
  // are required to be semantically equal, so multiple registrations for the
  // same op are allowed (making static initialization easier).
  void AddMetadataFn(string_view op_name, OpMetadataFn metadata_fn);

 private:
  friend class CpuOpHandler;
  CpuOpRegistry(const CpuOpRegistry&) = delete;
  CpuOpRegistry& operator=(const CpuOpRegistry&) = delete;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Use this macro to add a function that will register ops that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the tfrt::OpRegistration alias.
#define TFRT_STATIC_CPU_OP_REGISTRATION(FUNC) \
  TFRT_STATIC_CPU_OP_REGISTRATION_(FUNC, __COUNTER__)
#define TFRT_STATIC_CPU_OP_REGISTRATION_(FUNC, N) \
  TFRT_STATIC_CPU_OP_REGISTRATION__(FUNC, N)
#define TFRT_STATIC_CPU_OP_REGISTRATION__(FUNC, N)      \
  static bool tfrt_static_op_##N##_registered_ = []() { \
    ::tfrt::AddStaticCpuOpRegistration(FUNC);           \
    return true;                                        \
  }()

// The type for op registration functions. This is the same as the
// prototype for the entry point function for dynamic plugins.
using CpuOpRegistration = void (*)(CpuOpRegistry*);

// Adds an op to the registry. This should not be used directly; use
// TFRT_STATIC_CPU_OP_REGISTRATION instead.
void AddStaticCpuOpRegistration(CpuOpRegistration func);

// This is called to register all the statically linked ops in the given
// registry.
void RegisterStaticCpuOps(CpuOpRegistry* op_reg);

// Create a OpHandler from the specified registry of CPU ops and kernels.  The
// op_handler is owned by CoreRuntime, just like all op_handlers.  The fallback
// handler is used to run ops not registered with CPU op_handler.
Expected<OpHandler*> CreateCpuOpHandler(CoreRuntime* runtime,
                                        OpHandler* fallback_op_handler);
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_CORE_RUNTIME_CPU_OP_REGISTRY_H_
