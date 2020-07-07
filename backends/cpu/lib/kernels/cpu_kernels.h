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

//===- cpu_kernels.h --------------------------------------------*- C++ -*-===//
//
// This file defines cpu kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_

#include "mkldnn.h"  // from @mkl_dnn
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/msan.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"

namespace tfrt {
namespace cpu {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::EigenHostContext;
using ::tfrt::compat::KeepBuffers;

//===----------------------------------------------------------------------===//
// CPU Add kernels
//===----------------------------------------------------------------------===//

// This implementation is kept in order to support addition of two tensors with
// shapes [1, 3] and [3] as needed by cpp_tests/tpu_model:simple_model_test.
// TODO(b/160701946): Remove this non-Eigen based implementation.
template <typename T>
AsyncValueRef<HostTensor> AddSlow(const HostTensor& lhs_ref,
                                  const HostTensor& rhs_ref,
                                  const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto* lhs = &lhs_ref;
  auto* rhs = &rhs_ref;

  // We handle Scalar+Scalar, Scalar+Dense, Dense+Dense below. Swap
  // Dense+Scalar to simplify the logic since add is commutative.
  if (isa<DenseHostTensor>(lhs) && isa<AnyScalarHostTensor>(rhs))
    std::swap(lhs, rhs);

  // Handle scalar+scalar.
  if (auto* srhs = dyn_cast<ScalarHostTensor<T>>(rhs)) {
    auto* slhs = cast<ScalarHostTensor<T>>(lhs);
    auto result = slhs->GetValue() + srhs->GetValue();
    return host->template MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(
        slhs->metadata(), result);
  }

  auto dest = DenseHostTensor::CreateUninitialized(lhs->metadata(), host);
  if (!dest)
    return host->MakeErrorAsyncValueRef("out of memory allocating result");

  MutableDHTArrayView<T> dest_view(dest.getPointer());

  // Handle scalar+dense.
  DHTArrayView<T> rhs_view(cast<DenseHostTensor>(rhs));
  if (auto* slhs = dyn_cast<ScalarHostTensor<T>>(lhs)) {
    // Add a scalar to a dense tensor.
    auto lhs = slhs->GetValue();
    for (size_t i = 0, e = dest_view.NumElements(); i != e; ++i)
      dest_view[i] = lhs + rhs_view[i];
  } else {
    // Add two dense tensors.
    DHTArrayView<T> lhs_view(cast<DenseHostTensor>(lhs));
    for (size_t i = 0, e = dest_view.NumElements(); i != e; ++i)
      dest_view[i] = lhs_view[i] + rhs_view[i];
  }
  return host->MakeAvailableAsyncValueRef<DenseHostTensor>(
      std::move(dest.getValue()));
}

// TODO(fishx): Let this kernel support fp16.
// TODO(b/160701946): Update this kernel to support broadcasting.
template <typename T>
AsyncValueRef<HostTensor> Add(const HostTensor& lhs_ref,
                              const HostTensor& rhs_ref,
                              const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto* lhs = &lhs_ref;
  auto* rhs = &rhs_ref;

  if (lhs->shape() != rhs->shape() && lhs->dtype() == rhs->dtype() &&
      lhs->NumElements() == rhs->NumElements()) {
    // Fallback to the legacy AddSlow() to do broadcasting add.
    return AddSlow<T>(lhs_ref, rhs_ref, exec_ctx);
  }

  if (lhs->metadata() != rhs->metadata()) {
    return EmitErrorAsync(exec_ctx, "lhs metadata does not match rhs metadata");
  }

  // Handle scalar+scalar, scalar+dense, dense+dense below. Swap
  // Dense+Scalar to simplify the logic since add is commutative.
  if (isa<DenseHostTensor>(lhs) && isa<AnyScalarHostTensor>(rhs))
    std::swap(lhs, rhs);

  // Handle scalar+scalar.
  if (auto* rhs_scalar = dyn_cast<ScalarHostTensor<T>>(rhs)) {
    auto* lhs_scalar = cast<ScalarHostTensor<T>>(lhs);
    auto result = lhs_scalar->GetValue() + rhs_scalar->GetValue();
    return host->template MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(
        lhs_scalar->metadata(), result);
  }

  auto output = DenseHostTensor::CreateUninitialized(rhs->metadata(), host);
  if (!output)
    return host->MakeErrorAsyncValueRef("out of memory allocating result");

  MutableDHTArrayView<T> output_view(output.getPointer());
  auto output_t = AsEigenTensor(output_view);
  auto result = host->MakeUnconstructedAsyncValueRef<DenseHostTensor>();

  AsyncValueRef<Chain> chain;
  auto* rhs_dht = cast<DenseHostTensor>(rhs);
  DHTArrayView<T> rhs_view(rhs_dht);
  auto rhs_t = AsEigenConstTensor(rhs_view);

  if (auto* lhs_scalar = dyn_cast<ScalarHostTensor<T>>(lhs)) {
    // Handle scalar+dense.
    auto lhs = lhs_scalar->GetValue();
    auto expr = rhs_t + lhs;

    chain = AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                        std::move(output_t), std::move(expr),
                        KeepBuffers::alive(rhs_dht, output.getPointer()));
  } else {
    // Handle dense+dense.
    auto* lhs_dht = cast<DenseHostTensor>(lhs);
    DHTArrayView<T> lhs_view(lhs_dht);
    auto lhs_t = AsEigenConstTensor(lhs_view);

    auto fn = [](auto& a, auto& b, auto& c) { return a + b; };
    auto expr = fn(lhs_t, rhs_t, output_t);

    chain =
        AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                    std::move(output_t), std::move(expr),
                    KeepBuffers::alive(lhs_dht, rhs_dht, output.getPointer()));
  }

  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen([output = std::move(output).getValue(),
                     chain = std::move(chain),
                     result = result.CopyRef()]() mutable {
    if (chain.IsError()) {
      result.SetError(chain.GetError());
    } else {
      result.emplace(std::move(output));
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// CPU Matmul kernels
//===----------------------------------------------------------------------===//

template <typename T>
void MatMul2DKernel(T alpha, DHTIndexableView<T, 2> A, DHTIndexableView<T, 2> B,
                    T beta, MutableDHTIndexableView<T, 2>& C, bool transpose_a,
                    bool transpose_b) {
  T zero = static_cast<T>(0);
  // TODO(zhangqiaorjc): Handle transpose.
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

template <>
inline void MatMul2DKernel<float>(float alpha, DHTIndexableView<float, 2> A,
                                  DHTIndexableView<float, 2> B, float beta,
                                  MutableDHTIndexableView<float, 2>& C,
                                  bool transpose_a, bool transpose_b) {
  // MKL-DNN sgemm computes C = alpha * A @ B + beta * C, assuming all matrices
  // are column-major. MLIR tensors are row-major. We compute,
  //   C_rowmajor = C_colmajor^T = B_colmajor^T * A_colmajor^T,
  // feeding in B_rowmajor for B_colmajor^T and A_rowmajor for A_colmajor^T.
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

  // MKL DNN only supports the Fortran api and requires column major while we
  // use row major so we reverse the order A and B.
  mkldnn_status_t status =
      mkldnn_sgemm(&trans_b, &trans_a, &n, &m, &k, &alpha, B.data(), &ldb,
                   A.data(), &lda, &beta, C.data(), &ldc);
  assert(status == mkldnn_status_t::mkldnn_success);

  // assert is a no-op in optimized mode so we add this to avoid compiler's
  // unused-variable error.
  EIGEN_UNUSED_VARIABLE(status);

  // Since MKL is pre-built library, it causes "use-of-uninitialized-value" msan
  // warning.
  TFRT_MSAN_MEMORY_IS_INITIALIZED(C.data(), C.NumElements() * sizeof(float));
}

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

//===----------------------------------------------------------------------===//
// CPU Relu kernels
//===----------------------------------------------------------------------===//

// Computes B = Relu(A).
template <typename T>
static AsyncValueRef<Chain> Relu(const DenseHostTensor& A, DenseHostTensor* B,
                                 const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return a.cwiseMax(static_cast<T>(0)); };
  return ::tfrt::compat::UnaryEigenKernelAsync<T, T>(A, B, std::move(fn),
                                                     exec_ctx);
}

//===----------------------------------------------------------------------===//
// CPU Mean kernels
//===----------------------------------------------------------------------===//

template <typename T>
static AsyncValueRef<Chain> Mean(const DenseHostTensor& input,
                                 ArrayRef<int32_t> reduction_indices,
                                 DenseHostTensor* output,
                                 const ExecutionContext& exec_ctx) {
  // shape_input has format (batch_size, height, width, in_channel_num)
  DHTIndexableView<T, 4> input_view(&input);
  MutableDHTIndexableView<T, 2> output_view(output);
  const auto& shape_input = input_view.FixedShape();
  // shape_output has format (batch_size, in_channel_num)
  const auto& shape_output = output_view.FixedShape();

  const FixedRankShape<2> expected_output_shape(
      {shape_input[0], shape_input[3]});

  if (shape_output != expected_output_shape) {
    return EmitErrorAsync(exec_ctx, "unexpected output shape");
  }

  if (reduction_indices.size() != 2) {
    return EmitErrorAsync(exec_ctx,
                          "Only reduction indices with size 2 is supported");
  }

  Eigen::DSizes<Eigen::Index, 2> reduction_indices_t;
  reduction_indices_t[0] = reduction_indices[0];
  reduction_indices_t[1] = reduction_indices[1];
  auto input_t = AsEigenConstTensor(input_view);
  auto output_t = AsEigenTensor(output_view);
  auto expr = input_t.mean(reduction_indices_t);

  return AsyncAssign(
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
      std::move(output_t), std::move(expr), KeepBuffers::alive(&input, output));
}

//===----------------------------------------------------------------------===//
// CPU BiasAdd kernels
//===----------------------------------------------------------------------===//

template <typename T>
static AsyncValueRef<Chain> BiasAdd(const DenseHostTensor& input,
                                    const DenseHostTensor& bias,
                                    DenseHostTensor* output,
                                    const ExecutionContext& exec_ctx) {
  DHTIndexableView<T, 2> input_view(&input);
  DHTIndexableView<T, 1> bias_view(&bias);
  MutableDHTIndexableView<T, 2> output_view(output);
  const auto& shape_input = input_view.FixedShape();
  const auto& shape_bias = bias_view.FixedShape();
  const auto& shape_output = output_view.FixedShape();

  if (shape_input != shape_output) {
    return EmitErrorAsync(exec_ctx, "unexpected output shape");
  }

  if (shape_bias[0] != shape_input[1]) {
    return EmitErrorAsync(exec_ctx, "bias shape does not match input shape");
  }

  Eigen::array<Eigen::Index, 2> reshape_dims;
  reshape_dims[0] = static_cast<Eigen::Index>(1);
  reshape_dims[1] = static_cast<Eigen::Index>(shape_bias[0]);

  Eigen::array<Eigen::Index, 2> broadcast_dims;
  broadcast_dims[0] = static_cast<Eigen::Index>(shape_input[0]);
  broadcast_dims[1] = static_cast<Eigen::Index>(1);

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
