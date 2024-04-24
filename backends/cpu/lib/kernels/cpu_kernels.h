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
#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <type_traits>

#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/diagnostic.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/msan.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace cpu {

using ::Eigen::Index;
using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::EigenHostContext;
using ::tfrt::compat::KeepBuffers;

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
                     HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.cwiseMax(static_cast<T>(0)); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
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

//===----------------------------------------------------------------------===//
// CPU Sigmoid kernels
//===----------------------------------------------------------------------===//

// Computes B = Sigmoid(A) in sync style.
template <typename T>
llvm::Error SyncSigmoid(const DenseHostTensor& A, DenseHostTensor* B,
                        HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.sigmoid(); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
}

//===----------------------------------------------------------------------===//
// CPU Sum kernels
//===----------------------------------------------------------------------===//

// Computes B = Sum(A) in sync style for 2D tensors (only).
template <typename T>
llvm::Error SyncSum2D(const DenseHostTensor& A, DenseHostTensor* B, int axis) {
  if (A.shape().GetRank() != 2 || B->shape().GetRank() != 2) {
    return MakeStringError("Inputs must be rank-2 tensors.");
  }
  using EigenMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const EigenMatrix> input(A.data<T>(),
                                      A.shape().GetDimensionSize(0),
                                      A.shape().GetDimensionSize(1));
  Eigen::Map<EigenMatrix> output(B->data<T>(), B->shape().GetDimensionSize(0),
                                 B->shape().GetDimensionSize(1));
  if (axis == 0) {
    output = input.colwise().sum();
  } else {
    output = input.rowwise().sum();
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// CPU Exp kernels
//===----------------------------------------------------------------------===//

// Computes B = Exp(A) in sync style.
template <typename T>
llvm::Error SyncExp(const DenseHostTensor& A, DenseHostTensor* B,
                    HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.exp(); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
}

//===----------------------------------------------------------------------===//
// CPU comparison kernels
//===----------------------------------------------------------------------===//

// Computes c = Greater(A, B) in sync style.
template <typename T>
llvm::Error SyncGreater(const DenseHostTensor& A, const DenseHostTensor& B,
                        DenseHostTensor* C, HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b, auto& c) { return a > b; };
  if (B.shape().GetRank() == 0) {
    return ::tfrt::compat::BinaryEigenKernelBroadcast<T, bool>(
        A, B.data<T>()[0], C, std::move(fn), host_ctx);
  } else {
    return ::tfrt::compat::BinaryEigenKernel<T, bool>(A, B, C, std::move(fn),
                                                      host_ctx);
  }
}

// Computes c = Min(A, B) in sync style.
template <typename T>
llvm::Error SyncMinimum(const DenseHostTensor& A, const DenseHostTensor& B,
                        DenseHostTensor* C, HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b, auto& c) { return a.cwiseMin(b); };

  if (B.shape().GetRank() == 0) {
    return ::tfrt::compat::BinaryEigenKernelBroadcast<T, T>(
        A, B.data<T>()[0], C, std::move(fn), host_ctx);
  } else {
    return ::tfrt::compat::BinaryEigenKernel<T, T>(A, B, C, std::move(fn),
                                                   host_ctx);
  }
}

// Computes c = Max(A, B) in sync style.
template <typename T>
llvm::Error SyncMaximum(const DenseHostTensor& A, const DenseHostTensor& B,
                        DenseHostTensor* C, HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b, auto& c) { return a.cwiseMax(b); };

  if (B.shape().GetRank() == 0) {
    return ::tfrt::compat::BinaryEigenKernelBroadcast<T, T>(
        A, B.data<T>()[0], C, std::move(fn), host_ctx);
  } else {
    return ::tfrt::compat::BinaryEigenKernel<T, T>(A, B, C, std::move(fn),
                                                   host_ctx);
  }
}

template <typename T>
llvm::Error SyncSoftplus(const DenseHostTensor& input, DenseHostTensor* output,
                         HostContext& host_ctx) {
  auto fn = [](auto& a, auto& c) {
    static const T threshold =
        Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
    // Value above which exp(x) may overflow, but softplus(x) == x
    // is within machine epsilon.
    auto too_large = a > a.constant(-threshold);
    // Value below which exp(x) may underflow, but softplus(x) == exp(x)
    // is within machine epsilon.
    auto too_small = a < a.constant(threshold);
    auto features_exp = a.exp();
    return too_large.select(
        a,                              // softplus(x) ~= x for x large
        too_small.select(features_exp,  // softplus(x) ~= exp(x) for x small
                         features_exp.log1p()));
  };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(input, output, std::move(fn),
                                                host_ctx);
}

// Computes B = Sqrt(A) in sync style.
template <typename T>
llvm::Error SyncSqrt(const DenseHostTensor& A, DenseHostTensor* B,
                     HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.sqrt(); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
}

// Computes B = Mean(A) in sync style for 2D tensors (only). Note this supports
// reduction only along 1 axis for now
template <typename T, int NDIMS>
llvm::Error SyncMean2D(const DenseHostTensor& input,
                       Eigen::array<int, 1> reduction_axes,
                       DenseHostTensor* result, HostContext& host_ctx) {
  auto input_view = DHTIndexableView<T, NDIMS>(
      input.data<T>(), FixedRankShape<NDIMS>(input.shape()));

  auto result_view = MutableDHTArrayView<T>(result);

  auto a_tensor_eigen = AsEigenConstTensor(input_view);
  auto result_tensor_eigen = AsEigenTensor(result_view);
  Eigen::internal::SumReducer<T> sum_reducer;
  result_tensor_eigen.device(
      host_ctx.GetOrCreateSharedContext<EigenHostContext>().Device()) =
      a_tensor_eigen.reduce(reduction_axes, sum_reducer) /
      static_cast<T>(a_tensor_eigen.size() / result_tensor_eigen.size());

  return llvm::Error::success();
}

// Computes B = Log1p(A) in sync style.
template <typename T>
llvm::Error SyncLog1p(const DenseHostTensor& A, DenseHostTensor* B,
                      HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.log1p(); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
}

// Computes B = Tanh(A) in sync style.
template <typename T>
llvm::Error SyncTanh(const DenseHostTensor& A, DenseHostTensor* B,
                     HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.tanh(); };
  return ::tfrt::compat::UnaryEigenKernel<T, T>(A, B, std::move(fn), host_ctx);
}

// Computes B = Cast(A) in sync style.
template <typename SrcT, typename DstT>
llvm::Error SyncCast(const DenseHostTensor& A, DenseHostTensor* B,
                     HostContext& host_ctx) {
  auto fn = [](auto& a, auto& b) { return a.template cast<DstT>(); };
  return ::tfrt::compat::UnaryEigenKernel<SrcT, DstT>(A, B, std::move(fn),
                                                      host_ctx);
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CPU_KERNELS_H_
