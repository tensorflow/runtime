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

//===- results.cc - -------------------------------------------------------===//
// Returning results from compiled executables to the caller.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/results.h"

#include <utility>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/utils/async_runtime_api.h"
#include "third_party/tensorflow/compiler/xla/runtime/types.h"

namespace tfrt {
namespace jitrt {

using xla::runtime::AsyncTokenType;
using xla::runtime::ConvertAsyncTokenToChain;
using xla::runtime::Type;

namespace {
// Do not record any operands information for results conversion.
struct ConversionCtx {};

template <typename T, int rank>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
  return llvm::ArrayRef(memref->sizes);
}

template <typename T>
static ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
  return {};
}

// The returned memref can point into statically allocated memory that we can't
// pass to `free` (memref.global). The LLVM lowering of `memref.global` sets the
// allocated pointer to the magic value 0xDEADBEEF.
template <typename T, int rank>
static bool IsStaticStorageDuration(StridedMemRefType<T, rank>* memref) {
  return reinterpret_cast<std::intptr_t>(memref->basePtr) == 0xDEADBEEF;
}

// Converts StridedMemref to the DenseHostTensor. This struct satisfies
// ReturnStridedMemref's concept (see jitrt.h).
//
// This converter always creates a new DenseHostTensor from the memref, and it
// must be used only when it is guaranteed that the compiled region can't
// return global constant memref or forward one of the operands.
struct ConvertDenseHostTensor {
  using ResultType = DenseHostTensor;
  using ConversionContext = ConversionCtx;

  template <typename T, int rank>
  static DenseHostTensor Convert(ConversionContext& ctx, void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    TFRT_MSAN_MEMORY_IS_INITIALIZED(memref, sizeof(StridedMemRefType<T, rank>));
    TensorMetadata metadata(GetDType<T>(), Sizes(memref));
    TFRT_MSAN_MEMORY_IS_INITIALIZED(memref->data,
                                    metadata.GetHostSizeInBytes());

    // Deallocate memref only if it has dynamic storage duration.
    void* ptr = IsStaticStorageDuration(memref) ? nullptr : memref->basePtr;
    HostBuffer::Deallocator deallocator = [ptr](void*, size_t) { free(ptr); };

    return DenseHostTensor(
        metadata, HostBuffer::CreateFromExternal(memref->data,
                                                 metadata.GetHostSizeInBytes(),
                                                 std::move(deallocator)));
  }
};
}  // namespace

namespace internal {

mlir::LogicalResult ReturnAsyncToken(RemainingResults results,
                                     unsigned result_index, const Type* type,
                                     const Type* runtime_type,
                                     void* result_ptr) {
  if (!isa<AsyncTokenType>(type)) return mlir::failure();

  // Load the pointer to the async token from a pointer to result storage.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(result_ptr, sizeof(void*));
  void* ret = *reinterpret_cast<void**>(result_ptr);
  auto* token = static_cast<mlir::runtime::AsyncToken*>(ret);
  results[result_index] = ConvertAsyncTokenToChain(token);
  return mlir::success();
}

mlir::LogicalResult ReturnAsyncMemrefAsDenseHostTensor(RemainingResults results,
                                                       unsigned result_index,
                                                       const Type* type,
                                                       const Type* runtime_type,
                                                       void* result_ptr) {
  ConversionCtx ctx;
  return ReturnAsyncStridedMemref<ConvertDenseHostTensor>(
      ctx, results, result_index, type, runtime_type, result_ptr);
}

mlir::LogicalResult ReturnMemrefAsDenseHostTensor(RemainingResults results,
                                                  unsigned result_index,
                                                  const Type* type,
                                                  const Type* runtime_type,
                                                  void* result_ptr) {
  ConversionCtx ctx;
  return ReturnStridedMemref<ConvertDenseHostTensor>(
      ctx, results, result_index, type, runtime_type, result_ptr);
}

}  // namespace internal

void ReturnErrors(RemainingResults results, Error error) {
  auto async_error = MakeErrorAsyncValueRef(StrCat(error));
  for (int i = 0; i < results.size(); ++i) results[i] = async_error;
}

void ReturnErrors(RemainingResults results, DecodedDiagnostic error) {
  return ReturnErrors(results, MakeStringError(error));
}

}  // namespace jitrt
}  // namespace tfrt
