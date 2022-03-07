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

// This file defines cpu kernels.

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_

#include <type_traits>

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/msan.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"

#ifdef __x86_64__
#include "dnnl.h"  // from @dnnl
#endif

namespace tfrt {
namespace cpu {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::EigenHostContext;
using ::tfrt::compat::KeepBuffers;

//===----------------------------------------------------------------------===//
// CPU Matmul kernels
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Remove these MatMul kernels (requires mnist kernels update).

template <typename T>
void MatMul2DKernel(T alpha, DHTIndexableView<T, 2> A, DHTIndexableView<T, 2> B,
                    T beta, MutableDHTIndexableView<T, 2>& C, bool transpose_a,
                    bool transpose_b) {
  T zero = static_cast<T>(0);
  // TODO(tfrt-dev): Handle transpose.
  assert(transpose_a == false);
  assert(transpose_b == false);
  size_t rows = C.FixedShape()[0];
  size_t cols = C.FixedShape()[1];
  size_t inner_dim = A.FixedShape()[1];
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      T sum = zero;
      for (size_t k = 0; k < inner_dim; ++k) {
        sum += A.ElementAt(i, k) * B.ElementAt(k, j);
      }
      // We need to handle beta=0 without using C as input since C is
      // uninitialized to avoid msan errors.
      if (beta == zero)
        C.ElementAt(i, j) = alpha * sum;
      else
        C.ElementAt(i, j) = alpha * sum + beta * C.ElementAt(i, j);
    }
  }
}

#ifdef __x86_64__
template <>
inline void MatMul2DKernel<float>(float alpha, DHTIndexableView<float, 2> A,
                                  DHTIndexableView<float, 2> B, float beta,
                                  MutableDHTIndexableView<float, 2>& C,
                                  bool transpose_a, bool transpose_b) {
  // DNNL sgemm computes C = alpha * A @ B + beta * C, assuming all matrices
  // are row-major.
  // TODO(penporn): Support column-major when MLIR has column-major.

  // trans_a = 'N' or 'n',  op( A ) = A.
  // trans_a = 'T' or 't',  op( A ) = A**T.
  // trans_a = 'C' or 'c',  op( A ) = A**T.
  char trans_a = transpose_a ? 'T' : 'N';
  char trans_b = transpose_b ? 'T' : 'N';
  std::array<int, 2> dim_pair;
  dim_pair[0] = transpose_a ? 0 : 1;
  dim_pair[1] = transpose_b ? 1 : 0;

  const auto& shape_A = A.FixedShape();
  const auto& shape_B = B.FixedShape();
  assert(shape_A[dim_pair[0]] == shape_B[dim_pair[1]] &&
         "matmul arguments have incompatible shapes");

  // m: Specifies the number of rows of the matrix op(a) and of the matrix c.
  // The value of m must be at least zero.
  //
  // n: Specifies the number of columns of the matrix op(b) and the number of
  // columns of the matrix c. The value of n must be at least zero.
  //
  // k: Specifies the number of columns of the matrix op(a) and the number of
  // rows of the matrix op(b)
  int m = shape_A[1 - dim_pair[0]];
  int k = shape_A[dim_pair[0]];
  int n = shape_B[1 - dim_pair[1]];
  assert(m >= 0 && n >= 0 && k >= 0);

  // lda: Leading dimension of 'a' matrix. This is set at calling site depending
  // on transa parameter. Since DHT uses row-major layout, leading dimension is
  // the stride between consecutive rows lda = max(1,k) when transa is false,
  // otherwise lda = max(1,m)
  //
  // ldb: Leading dimension of 'b' matrix. This is set at calling site depending
  // on transb parameter. Since DHT uses row-major layout, leading dimension is
  // the stride between consecutive rows ldb = max(1,n) when transb is false,
  // otherwise ldb = max(1,k)
  //
  // ldc: Leading dimension of 'c' matrix. Since DHT uses row-major layout,
  // leading dimension is the stride between consecutive rows, max(1,n)
  int lda = transpose_a ? m : k;
  int ldb = transpose_b ? k : n;
  int ldc = n;

  dnnl_status_t status = dnnl_sgemm(trans_a, trans_b, m, n, k, alpha, A.data(),
                                    lda, B.data(), ldb, beta, C.data(), ldc);
  assert(status == dnnl_status_t::dnnl_success);

  // assert is a no-op in optimized mode so we add this to avoid compiler's
  // unused-variable error.
  EIGEN_UNUSED_VARIABLE(status);

  // Since MKL is pre-built library, it causes "use-of-uninitialized-value" msan
  // warning.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(C.data(), C.NumElements() * sizeof(float));
}
#endif  // __x86_64__

// TODO(tfrt-devs): Merge this into the matmul kernel interface layer, or
// expose alpha/beta as attributes.  Either this extensibility is important or
// it is not, we should pick :-).
template <typename T>
void CallMatMulKernel(const DenseHostTensor& lhs_dht,
                      const DenseHostTensor& rhs_dht, DenseHostTensor* dest_dht,
                      bool transpose_a, bool transpose_b) {
  DHTIndexableView<T, 2> lhs(&lhs_dht);
  DHTIndexableView<T, 2> rhs(&rhs_dht);
  MutableDHTIndexableView<T, 2> dest(dest_dht);

  MatMul2DKernel<T>(/*alpha=*/static_cast<T>(1), lhs, rhs,
                    /*beta=*/static_cast<T>(0), dest, transpose_a, transpose_b);
}

// This is the MatMul kernel interface.
// TODO(rmlarsen): Add support for transposition and conjugation.
// TODO(haoliang): Unify `MatMul2D` and `SyncMatMul2D` with a template function.
template <typename T>
Expected<Chain> MatMul2D(T alpha, DHTIndexableView<T, 2> A,
                         DHTIndexableView<T, 2> B, T beta,
                         MutableDHTIndexableView<T, 2> C) {
  // TODO(rmlarsen): Add support for transposition and conjugation.
  bool transpose_a = false;
  bool transpose_b = false;
  MatMul2DKernel<T>(alpha, A, B, beta, C, transpose_a, transpose_b);
  return Chain();
}

// Computes MatMul in sync style.
template <typename T>
void SyncMatMul2D(T alpha, DHTIndexableView<T, 2> A, DHTIndexableView<T, 2> B,
                  T beta, MutableDHTIndexableView<T, 2> C) {
  // TODO(rmlarsen): Add support for transposition and conjugation.
  bool transpose_a = false;
  bool transpose_b = false;
  MatMul2DKernel<T>(alpha, A, B, beta, C, transpose_a, transpose_b);
}

//===----------------------------------------------------------------------===//
// CPU Relu kernels
//===----------------------------------------------------------------------===//

// Computes B = Relu(A).
// TODO(haoliang): Unify `Relu` and `SyncRelu` with a template function.
template <typename T>
AsyncValueRef<Chain> Relu(const DenseHostTensor& A, DenseHostTensor* B,
                          const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return a.cwiseMax(static_cast<T>(0)); };
  return ::tfrt::compat::UnaryEigenKernelAsync<T, T>(A, B, std::move(fn),
                                                     exec_ctx);
}

// Computes B = Relu(A) in sync style.
template <typename T>
llvm::Error SyncRelu(const DenseHostTensor& A, DenseHostTensor* B,
                     const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return a.cwiseMax(static_cast<T>(0)); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), exec_ctx);
}

//===----------------------------------------------------------------------===//
// CPU Mean kernels
//===----------------------------------------------------------------------===//

template <typename T>
AsyncValueRef<Chain> Mean(const DenseHostTensor& input,
                          ArrayRef<int32_t> reduction_indices,
                          DenseHostTensor* output,
                          const ExecutionContext& exec_ctx) {
  auto mean = [&](auto input_rank_tag,
                  auto reduction_rank_tag) -> AsyncValueRef<Chain> {
    constexpr int input_rank = decltype(input_rank_tag)::value;
    constexpr int reduction_rank = decltype(reduction_rank_tag)::value;
    constexpr int output_rank = input_rank - reduction_rank;
    static_assert(output_rank >= 0, "Output rank must be greater than 0");

    DHTIndexableView<T, input_rank> input_view(&input);
    MutableDHTIndexableView<T, output_rank> output_view(output);

    Eigen::DSizes<Eigen::Index, reduction_rank> reduction_indices_t;
    for (int i = 0; i < reduction_rank; ++i) {
      reduction_indices_t[i] = reduction_indices[i];
    }

    auto input_t = AsEigenConstTensor(input_view);
    auto output_t = AsEigenTensor(output_view);
    auto expr = input_t.mean(reduction_indices_t);

    return AsyncAssign(
        exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
        std::move(output_t), std::move(expr),
        KeepBuffers::alive(&input, output));
  };

  const int input_rank = input.shape().GetRank();
  const int reduction_rank = reduction_indices.size();

#define REDUCE(INPUT, REDUCTION)                          \
  if (input_rank == INPUT && reduction_rank == REDUCTION) \
    return mean(std::integral_constant<int32_t, INPUT>{}, \
                std::integral_constant<int32_t, REDUCTION>{});

  REDUCE(1, 1);
  REDUCE(2, 1);
  REDUCE(2, 2);
  REDUCE(3, 1);
  REDUCE(3, 2);
  REDUCE(3, 3);
  REDUCE(4, 1);
  REDUCE(4, 2);
  REDUCE(4, 3);
  REDUCE(4, 4);

#undef REDUCE

  return EmitErrorAsync(
      exec_ctx, StrCat("Unsupported reduction ranks: input_rank=", input_rank,
                       " reduction_rank=", reduction_rank));
}

//===----------------------------------------------------------------------===//
// CPU BiasAdd kernels
//===----------------------------------------------------------------------===//

// A special case of tf.add where bias is restricted to be 1-D.
// Currently only support NHWC data format.
template <typename T, size_t RANK>
AsyncValueRef<Chain> BiasAdd(const DenseHostTensor& input,
                             const DenseHostTensor& bias,
                             DenseHostTensor* output,
                             const ExecutionContext& exec_ctx) {
  DHTIndexableView<T, RANK> input_view(&input);
  MutableDHTIndexableView<T, RANK> output_view(output);
  DHTIndexableView<T, 1> bias_view(&bias);

  const auto& shape_input = input_view.FixedShape();
  const auto& shape_bias = bias_view.FixedShape();
  const auto& shape_output = output_view.FixedShape();

  if (shape_input != shape_output) {
    return EmitErrorAsync(exec_ctx, "unexpected output shape");
  }

  if (shape_bias[0] != shape_input[RANK - 1]) {
    return EmitErrorAsync(exec_ctx, "bias shape does not match input shape");
  }

  // Reshape bias to the shape of input. Broadcast along the last axis of input.
  Eigen::array<Eigen::Index, RANK> reshape_dims;
  Eigen::array<Eigen::Index, RANK> broadcast_dims;
  for (size_t i = 0; i < RANK - 1; ++i) {
    reshape_dims[i] = static_cast<Eigen::Index>(1);
    broadcast_dims[i] = static_cast<Eigen::Index>(shape_input[i]);
  }
  reshape_dims[RANK - 1] = static_cast<Eigen::Index>(shape_bias[0]);
  broadcast_dims[RANK - 1] = static_cast<Eigen::Index>(1);

  auto input_t = AsEigenConstTensor(input_view);
  auto bias_t = AsEigenConstTensor(bias_view);
  auto output_t = AsEigenTensor(output_view);
  auto expr = input_t + bias_t.reshape(reshape_dims).broadcast(broadcast_dims);

  return AsyncAssign(
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
      std::move(output_t), std::move(expr),
      KeepBuffers::alive(&input, &bias, output));
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_
