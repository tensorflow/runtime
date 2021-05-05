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

// This file defines helper functions for writing Eigen-based kernels.

#ifndef TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNEL_H_
#define TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNEL_H_

#include <functional>

#include "llvm/Support/Error.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/common/compat/eigen/thread_pool_device.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace compat {

#define TFRT_RETURN_IF_ERROR(KERNEL_ERROR, ERR)               \
  do {                                                        \
    if (auto err = ERR) {                                     \
      llvm::Error unknown = llvm::handleErrors(               \
          std::move(err), [&](const llvm::StringError& err) { \
            KERNEL_ERROR.ReportError(err.getMessage());       \
          });                                                 \
      assert(!unknown && "Unknown error type");               \
      return;                                                 \
    }                                                         \
  } while (0)

// Helper function for implementing synchronous Eigen-based nullary kernels with
// matching input and output shapes. The functor Fn should return an Eigen
// tensor expression that can be assigned to an object of type
// EigenTensor<T, 1>.
template <typename T, typename Fn>
void NullaryEigenKernel(
    // `argument` supplies the buffer for both input and output.
    MutableDHTArrayView<T> argument, Fn fn, Location loc) {
  auto inout = AsEigenTensor(argument);
  inout = fn(inout);
}

// Helper function for implementing asynchronous Eigen-based nullary kernels.
// The functor Fn should return an Eigen tensor expression that can be assigned
// to an object of type EigenTensor<T, 1>.
template <typename T, typename Fn>
AsyncValueRef<Chain> NullaryEigenKernelAsync(
    // `argument` supplies the buffer for both input and output.
    DenseHostTensor* argument, Fn fn, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto argument_view = MutableDHTArrayView<T>(argument);
  auto inout = AsEigenTensor(argument_view);
  auto expr = fn(inout);
  // Execute the Eigen computation "inout = fn(inout);" asynchronously.
  return AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                     std::move(inout), std::move(expr),
                     KeepBuffers::alive(argument));
}

// Implements synchronous Eigen-based unary kernels with
// matching input and output shapes. The functor Fn should return an Eigen
// tensor expression that can be assigned to an object of type
// EigenTensor<Tout, 1>.
template <typename Tin, typename Tout, typename Fn>
llvm::Error UnaryEigenKernel(
    const DenseHostTensor& input,
    // `output` supplies the buffer in which to write the output.
    DenseHostTensor* output, Fn fn, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto input_view = DHTArrayView<Tin>(&input);
  auto output_view = MutableDHTArrayView<Tout>(output);
  const TensorShape& shape_input = input.shape();
  const TensorShape& shape_output = output->shape();
  if (shape_input != shape_output) {
    return MakeStringError(" tensor shape mismatch: ", shape_input, " vs. ",
                           shape_output);
  }
  auto in = AsEigenConstTensor(input_view);
  auto out = AsEigenTensor(output_view);
  auto expr = fn(in, out);
  out.device(host->GetOrCreateSharedContext<EigenHostContext>().Device()) =
      expr;
  return llvm::Error::success();
}

// Helper function for implementing asynchronous Eigen-based unary kernels with
// matching input and output shapes. The functor Fn should return an Eigen
// tensor expression that can be assigned to an object of type
// EigenTensor<Tout, 1>. Returns the out chain.
template <typename Tin, typename Tout, typename Fn>
AsyncValueRef<Chain> UnaryEigenKernelAsync(
    const DenseHostTensor& input,
    // `output` supplies the buffer in which to write the output.
    DenseHostTensor* output, Fn fn, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto input_view = DHTArrayView<Tin>(&input);
  auto output_view = MutableDHTArrayView<Tout>(output);
  const auto& shape_input = input.metadata().shape;
  const auto& shape_output = output->metadata().shape;

  if (shape_input != shape_output) {
    return EmitErrorAsync(
        exec_ctx,
        StrCat("tensor shape mismatch: ", shape_input, " vs. ", shape_output));
  }

  auto in = AsEigenConstTensor(input_view);
  auto out = AsEigenTensor(output_view);
  auto expr = fn(in, out);

  return AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                     std::move(out), std::move(expr),
                     KeepBuffers::alive(&input, output));
}

// Helper function for implementing synchronous Eigen-based binary kernels with
// matching input and output shapes. The functor Fn should return an Eigen
// tensor expression that can be assigned to an object of type
// EigenTensor<Tout, Rank>.
template <typename Tin, typename Tout, typename Fn>
Expected<Chain> BinaryEigenKernel(
    DHTArrayView<Tin> left, DHTArrayView<Tin> right,
    // `output` supplies the buffer in which to write the output.
    MutableDHTArrayView<Tout>& output, Fn fn) {
  const auto& shape_left = left.Shape();
  const auto& shape_right = right.Shape();
  const auto& shape_output = output.Shape();
  if (shape_left != shape_right || shape_left != shape_output) {
    return MakeStringError(" tensor shape mismatch: ", shape_left, " vs. ",
                           shape_right, " vs. ", shape_output);
  }
  auto lhs = AsEigenConstTensor(left);
  auto rhs = AsEigenConstTensor(right);
  auto out = AsEigenTensor(output);
  out = fn(lhs, rhs, out);
  return Chain();
}

// Helper function for implementing asynchronous Eigen-based binary kernels with
// matching input and output shapes. The functor Fn should return an Eigen
// tensor expression that can be assigned to an object of type
// EigenTensor<Tout, Rank>.
template <typename Tin, typename Tout, typename Fn>
AsyncValueRef<Chain> BinaryEigenKernelAsync(
    const DenseHostTensor& left, const DenseHostTensor& right,
    // `output` supplies the buffer in which to write the output.
    DenseHostTensor* output, Fn fn, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto left_view = DHTArrayView<Tin>(&left);
  auto right_view = DHTArrayView<Tin>(&right);
  auto output_view = MutableDHTArrayView<Tout>(output);
  const auto& shape_left = left.metadata().shape;
  const auto& shape_right = right.metadata().shape;
  const auto& shape_output = output->metadata().shape;

  if (shape_left != shape_right || shape_left != shape_output) {
    return EmitErrorAsync(exec_ctx,
                          StrCat("tensor shape mismatch: ", shape_left, " vs. ",
                                 shape_right, " vs. ", shape_output));
  }
  auto lhs = AsEigenConstTensor(left_view);
  auto rhs = AsEigenConstTensor(right_view);
  auto out = AsEigenTensor(output_view);
  auto expr = fn(lhs, rhs, out);

  return AsyncAssign(host->GetOrCreateSharedContext<EigenHostContext>(),
                     std::move(out), std::move(expr),
                     KeepBuffers::alive(&left, &right, output));
}

}  // namespace compat
}  // namespace tfrt

#endif  // TFRT_BACKENDS_COMMON_COMPAT_EIGEN_KERNEL_H_
