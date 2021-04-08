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

// Helpers for gpu op implementations
//
// This file declares simple helper routines to make it easier to write
// gpu dispatch function for a op. This is intended to be small and
// simple things and is nearly header-only.

#ifndef TFRT_GPU_CORE_RUNTIME_GPU_OP_UTILS_H_
#define TFRT_GPU_CORE_RUNTIME_GPU_OP_UTILS_H_

#include <tuple>

#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/op_args.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"

namespace tfrt {

class OpAttrsRef;
class Tensor;
class TensorMetadata;

//===----------------------------------------------------------------------===//
// Registration helpers used to make GPU dispatch function easier to define.
//===----------------------------------------------------------------------===//

// TFRT_GPU_OP is a macro that makes defining dispatch function more
// straightforward.
//
// Example op definition:
//
// AddDispatchFn(GpuDispatchContext* dispatch_context,
//               const DenseGpuTensor& a,
//               const DenseGpuTensor& b,
//               const TensorMetadata& c_md,
//               Location loc) { ... }
//
// Example for dispatch function that needs OpAttrs:
//
// AddDispatchFn(GpuDispatchContext* dispatch_context,
//               const DenseGpuTensor& a,
//               const DenseGpuTensor& b,
//               const OpAttrsRef& attrs,
//               const TensorMetadata& c_md,
//               Location loc) { ... }
//
// We can wrap the above op definition using TFRT_GPU_OP when registering it
// with the op registry. Example:
//
//  registry->AddOp("tfrt_test.gpu_add", TFRT_GPU_OP(AddDispatchFn));

#define TFRT_GPU_OP(...)                                             \
  ::tfrt::DispatchFnImpl<GpuDispatchContext, decltype(&__VA_ARGS__), \
                         &__VA_ARGS__>::Invoke
}  // namespace tfrt
#endif  // TFRT_GPU_CORE_RUNTIME_GPU_OP_UTILS_H_
