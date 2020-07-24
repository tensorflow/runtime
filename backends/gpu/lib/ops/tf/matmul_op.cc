// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- matmul_op.cc - Implements tf.matmul on GPU --------------*- C++ -*--===//
//
//
//===----------------------------------------------------------------------===//
#include "matmul_op.h"

#include <immintrin.h>

#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/blas_support.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/fp16.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace gpu {

namespace {
template <typename T>
class ConstValue {
 public:
  explicit ConstValue(double value) : value_(static_cast<T>(value)) {}
  auto pointer(stream::Platform platform) const {
    return stream::Pointer<const T>(&value_, platform);
  }

 private:
  T value_;
};

// Specialization for incomplete type __half with storage type fp16.
template <>
class ConstValue<__half> {
 public:
  explicit ConstValue(double value)
      : value_(
            _cvtss_sh(static_cast<float>(value), _MM_FROUND_TO_NEAREST_INT)) {}
  auto pointer(stream::Platform platform) const {
    return stream::Pointer<const __half>(
        reinterpret_cast<const __half*>(&value_), platform);
  }

 private:
  fp16 value_;
};
}  // namespace

template <typename T>
static llvm::Error CallCublasGemm(stream::CurrentContext current,
                                  stream::BlasHandle handle, bool transpose_a,
                                  bool transpose_b, uint64_t m, uint64_t k,
                                  uint64_t n, const gpu::DenseGpuTensor& a,
                                  const gpu::DenseGpuTensor& b,
                                  GpuBuffer* result) {
  TFRT_TRACE_SCOPE("CublasGemm");
  // Blas expects matrices in column major.
  // Use C' = B' x A' (' stands for transpose)
  // clang-format off
  return CublasGemm(current, handle,
                    transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                    n, m, k,
                    ConstValue<T>(1.0).pointer(handle.platform()),
                    static_cast<stream::Pointer<const T>>(b.buffer().pointer()), transpose_b ? k : n,
                    static_cast<stream::Pointer<const T>>(a.buffer().pointer()), transpose_a ? m : k,
                    ConstValue<T>(0.0).pointer(handle.platform()),
                    static_cast<stream::Pointer<T>>(result->pointer()), n);
  // clang-format on
}

llvm::Error RunCublasGemm(stream::CurrentContext current,
                          stream::BlasHandle handle, bool transpose_a,
                          bool transpose_b, const gpu::DenseGpuTensor& a,
                          const gpu::DenseGpuTensor& b, GpuBuffer* result) {
  const int a_matching_dim = transpose_a ? 0 : 1;
  const int b_matching_dim = transpose_b ? 1 : 0;
  const int a_remaining_dim = 1 - a_matching_dim;
  const int b_remaining_dim = 1 - b_matching_dim;
  const uint64_t m = a.shape().GetDimensionSize(a_remaining_dim);
  const uint64_t k = a.shape().GetDimensionSize(a_matching_dim);
  const uint64_t n = b.shape().GetDimensionSize(b_remaining_dim);
  switch (a.dtype().kind()) {
    case DType::F16:
      return CallCublasGemm<__half>(current, handle, transpose_a, transpose_b,
                                    m, k, n, a, b, result);
    case DType::F32:
      return CallCublasGemm<float>(current, handle, transpose_a, transpose_b, m,
                                   k, n, a, b, result);
    case DType::F64:
      return CallCublasGemm<double>(current, handle, transpose_a, transpose_b,
                                    m, k, n, a, b, result);
    // TODO(iga): Handle complex numbers.
    default:
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          StrCat("Type ", a.dtype(), " is not supported by cuBLASS gemm"));
  }
}

static llvm::Expected<DenseGpuTensor> GpuMatmulOp(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& a,
    const gpu::DenseGpuTensor& b, const OpAttrsRef& attrs,
    const TensorMetadata& result_md) {
  TFRT_TRACE_SCOPE("GpuMatmulOp");

  size_t size_in_bytes = result_md.GetHostSizeInBytes();
  TFRT_ASSIGN_OR_RETURN(RCReference<GpuBuffer> buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  if (size_in_bytes == 0) {
    return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
  }

  if (a.shape().GetNumElements() == 0 && b.shape().GetNumElements() == 0) {
    // If a has shape [x, 0] and b has shape [0, y], the
    // output shape is [x, y] where x and y are non-zero, so we fill
    // the output with zeros.
    if (auto error =
            stream::MemsetD8Async(dctx->current_context(), buffer->pointer(), 0,
                                  size_in_bytes, dctx->stream())) {
      return std::move(error);
    }
    return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
  }

  // FIXME(iga): Handle types not supported by cuBLAS.

  // metadata function checks for attribute presence
  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");
  if (auto error =
          RunCublasGemm(dctx->current_context(), dctx->blas_handle(),
                        transpose_a, transpose_b, a, b, buffer.get())) {
    // TODO(iga): Propagate original error.
    return std::move(error);
  }

  return DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));
}

}  // namespace gpu

void RegisterMatmulGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.MatMul", TFRT_GPU_OP(gpu::GpuMatmulOp),
                  {"transpose_a", "transpose_b"});
}

}  // namespace tfrt
