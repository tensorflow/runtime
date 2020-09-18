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

//===- contraction_output_kernel.h ------------------------------*- C++ -*-===//
//
// Eigen Tensor contraction output kernel is a mechanism to fuse any
// element-wise operations into the Tensor contraction expression.
//
// See compat/eigen/contraction_output_kernel.h for the detailed documentation.
//
// Jit compiled output kernel provides a mechanism to fuse any computation
// into "the back of the Eigen contraction". Compiled output kernel allows
// to execute jit-compiled MLIR functions for each tensor contraction output
// block.
//
// In practice it allows to add computations for "free", because the data
// typically resides in the L1 cache.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_H_
#define TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_H_

#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {

class OpAttrsRef;

namespace cpu {
namespace jit {

// Forward declare types defined in .cc file.
class CompiledContractionOutputKernel;

// Returns contraction output kernel compiled from the MLIR function created
// by the contraction output kernel builder.
//
//   `dtype`           - data type of a contraction output
//   `additional_args` - data types of the additional arguments passed to the
//                       contraction output kernel (e.g. data type of the bias
//                       vector for BiasAdd output fusion)
Expected<CompiledContractionOutputKernel*> GetCompiledContractionOutputKernel(
    HostContext* host, ArrayRef<string_view> output_kernels,
    const OpAttrsRef& attrs, DType dtype, ArrayRef<DType> additional_args);

// Calls compiled output kernel for the contraction output block.
//
// Contraction output is a 2D array in the column major layout. Contraction
// output block is a view into that array at `row_offset` and `col_offset`
// offsets, of `rows`x`cols` size, and a `stride` in the inner dimension (size
// of contraction output inner dimension).
//
// The contraction output kernel can take arbitrary number of additional
// arguments (e.g. bias vector for BiasAdd output kernel). All additional
// arguments must be memrefs with statically known ranks, this is verified at
// the compile time.
void CallCompiledContractionOutputKernel(
    CompiledContractionOutputKernel* kernel, DType dtype, void* data,
    int64_t stride, int64_t row_offset, int64_t col_offset, int64_t rows,
    int64_t cols, ArrayRef<const DenseHostTensor*> additional_args);

// Verifies that additional output kernel arguments are compatible with compiled
// output kernel.
Error VerifyCompiledContractionOutoutKernelArgs(
    CompiledContractionOutputKernel* kernel, DType dtype,
    ArrayRef<const DenseHostTensor*> additional_args);

// Eigen contraction output kernel template that delegates all the work to the
// compiled output contraction kernel.
template <typename T>
class ContractionOutputKernel {
 public:
  using ContractionOutputMapper =
      Eigen::internal::blas_data_mapper<T, Eigen::Index, Eigen::ColMajor>;

  ContractionOutputKernel(CompiledContractionOutputKernel* kernel,
                          ArrayRef<const DenseHostTensor*> additional_args)
      : kernel_(kernel),
        additional_kernel_args_(additional_args.begin(),
                                additional_args.end()) {}

  void operator()(const ContractionOutputMapper& output_mapper,
                  const Eigen::TensorContractionParams& params,
                  Eigen::Index row_offset, Eigen::Index col_offset,
                  Eigen::Index rows, Eigen::Index cols) const {
    // In TFRT tensors are row-major and for row-major data layout Eigen swappes
    // lhs with rhs.
    assert(params.swapped_arguments);

    T* output_base = &output_mapper(0, 0);
    CallCompiledContractionOutputKernel(
        kernel_, GetDType<T>(), static_cast<void*>(output_base),
        output_mapper.stride(), row_offset, col_offset, rows, cols,
        additional_kernel_args_);
  }

 private:
  CompiledContractionOutputKernel* kernel_;
  SmallVector<const DenseHostTensor*, 8> additional_kernel_args_;
};

}  // namespace jit
}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_JIT_CONTRACTION_OUTPUT_KERNEL_H_
