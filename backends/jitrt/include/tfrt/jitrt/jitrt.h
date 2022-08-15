/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

// Support library for implementing TFRT kernels that do JIT compilation using
// MLIR framework (generating kernels at runtime from hight level MLIR
// dialects, e.g. generating dense linear algebra kernels from Linalg dialect).

#ifndef TFRT_BACKENDS_JITRT_JITRT_H_
#define TFRT_BACKENDS_JITRT_JITRT_H_

#include <sys/types.h>

#include <any>
#include <chrono>  // NOLINT(build/c++11)
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/jitrt/results.h"
#include "tfrt/support/forward_decls.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/calling_convention.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/compiler.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/specialization.h"
#include "third_party/tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_runtime.h"
#include "third_party/tensorflow/compiler/xla/runtime/async_values_cache.h"
#include "third_party/tensorflow/compiler/xla/runtime/constraints.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"
#include "third_party/tensorflow/compiler/xla/runtime/diagnostics.h"
#include "third_party/tensorflow/compiler/xla/runtime/executable.h"
#include "third_party/tensorflow/compiler/xla/runtime/execution_engine.h"
#include "third_party/tensorflow/compiler/xla/runtime/jit_executable.h"
#include "third_party/tensorflow/compiler/xla/runtime/memory_mapper.h"
#include "third_party/tensorflow/compiler/xla/runtime/symbolic_shape.h"
#include "third_party/tensorflow/compiler/xla/runtime/types.h"

// Forward declare Eigen types.
namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace mlir {
class PassManager;
}  // namespace mlir

// Forward declare types enabling compiled kernel <-> runtime integration.
namespace xla {
namespace runtime {
struct KernelContext;
}  // namespace runtime
}  // namespace xla

namespace tfrt {

class Tensor;

namespace jitrt {

// Converts tfrt Tensor to the Memref descriptor if concrete Tensor type is
// supported (currently only DenseHostTensor can be converted). Returns error
// otherwise.
Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor);

// Resource context caches all JitExecutables in the async value cache.
//
// We use compilation unit id as a cache key. Because this id is unique only
// within a single Bef file, it is the user's responsibility to guarantee that
// the JitExecutableCache is not reused between multiple Bef files.
using JitExecutableCache = AsyncValuesCache<size_t, JitExecutable>;

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_JITRT_H_
