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

//===- eigen_helper.cu.h - --------------------------------------*- C++ -*-===//
//
// This file has some utilities to implement ops using Eigen.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_GPU_LIB_OPS_TF_EIGEN_HELPER_CU_H_
#define TFRT_BACKENDS_GPU_LIB_OPS_TF_EIGEN_HELPER_CU_H_

#define EIGEN_USE_GPU

#include "llvm/ADT/FunctionExtras.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_shape.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h"  // from @eigen_archive

namespace tfrt {
namespace gpu {

// A type aggregate for packing up an Eigen functor signature. It starts with a
// result type, and then zero or more argument types. All types are scalar
// types.
template <typename ResultType, typename... ArgTypes>
struct FunctorSignature;

namespace internal {

template <typename T>
using AlignedEigenVector =
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, int32_t>,
                     Eigen::Aligned>;

template <typename FunctorSignature, typename Functor>
struct FunctorDispatch;

// Match a template with a single argument. This corresponds to a unary
// op.
template <typename ResultType, typename T, typename Functor>
struct FunctorDispatch<FunctorSignature<ResultType, T>, Functor> {
  static Error EvaluateFunctor(GpuDispatchContext* dctx,
                               const DenseGpuTensor& result_tensor,
                               ArrayRef<const DenseGpuTensor*> args) {
    assert(args.size() == 1);

    AlignedEigenVector<ResultType> result(
        GetRawPointer<ResultType>(result_tensor),
        result_tensor.shape().GetNumElements());
    AlignedEigenVector<T> arg(GetRawPointer<T>(*args[0]),
                              args[0]->shape().GetNumElements());

    assert(result.size() == arg.size());
    result.device(*dctx->eigen_gpu_device()) = arg.unaryExpr(Functor());

    return Error::success();
  }
};

// Match a template with a two arguments. This corresponds to a binary
// op.
template <typename ResultType, typename T0, typename T1, typename Functor>
struct FunctorDispatch<FunctorSignature<ResultType, T0, T1>, Functor> {
  static_assert(
      std::is_base_of<Eigen::internal::binary_op_base<T0, T1>, Functor>::value,
      "Binary op must derive from binary_op_base");

  using LhsType = typename Functor::first_argument_type;
  using RhsType = typename Functor::second_argument_type;

  // Handle as a broadcast case. result_tensor has the broadcasted shape.
  template <int rank>
  static void HandleAsBroadcast(GpuDispatchContext* dctx,
                                const DenseGpuTensor* lhs_tensor,
                                const DenseGpuTensor* rhs_tensor,
                                const DenseGpuTensor* result_tensor) {
    Eigen::array<int32_t, rank> lhs_shape;
    Eigen::array<int32_t, rank> rhs_shape;
    Eigen::array<int32_t, rank> result_shape;
    int num_lhs_added_dims = rank - lhs_tensor->shape().GetRank();
    int num_rhs_added_dims = rank - rhs_tensor->shape().GetRank();

    // broadcast by how many times. Eigen broadcast op requires the
    // multiplication factors (instead of the braodcasted shape) as input.
    Eigen::array<int32_t, rank> lhs_broadcast_factors;
    Eigen::array<int32_t, rank> rhs_broadcast_factors;
    for (int i = 0; i < rank; i++) {
      // Reshape lhs and rhs to the result rank. This involves padding 1 as the
      // leading dimensions until the ranks are all the same.
      lhs_shape[i] =
          i < num_lhs_added_dims
              ? 1
              : lhs_tensor->shape().GetDimensionSize(i - num_lhs_added_dims);
      rhs_shape[i] =
          i < num_rhs_added_dims
              ? 1
              : rhs_tensor->shape().GetDimensionSize(i - num_rhs_added_dims);
      result_shape[i] = result_tensor->shape().GetDimensionSize(i);

      assert(result_shape[i] % lhs_shape[i] == 0);
      lhs_broadcast_factors[i] = result_shape[i] / lhs_shape[i];

      assert(result_shape[i] % rhs_shape[i] == 0);
      rhs_broadcast_factors[i] = result_shape[i] / rhs_shape[i];
    }

    Eigen::TensorMap<Eigen::Tensor<LhsType, rank, Eigen::RowMajor, int32_t>,
                     Eigen::Aligned>
        lhs(GetRawPointer<LhsType>(*lhs_tensor), lhs_shape);
    Eigen::TensorMap<Eigen::Tensor<RhsType, rank, Eigen::RowMajor, int32_t>,
                     Eigen::Aligned>
        rhs(GetRawPointer<RhsType>(*rhs_tensor), rhs_shape);
    Eigen::TensorMap<Eigen::Tensor<ResultType, rank, Eigen::RowMajor, int32_t>,
                     Eigen::Aligned>
        result(GetRawPointer<ResultType>(*result_tensor), result_shape);

    result.device(*dctx->eigen_gpu_device()) =
        lhs.broadcast(lhs_broadcast_factors)
            .binaryExpr(rhs.broadcast(rhs_broadcast_factors), Functor());
  }

  static Error EvaluateFunctor(GpuDispatchContext* dctx,
                               const DenseGpuTensor& result_tensor,
                               ArrayRef<const DenseGpuTensor*> args) {
    assert(args.size() == 2);

    AlignedEigenVector<ResultType> result(
        GetRawPointer<ResultType>(result_tensor),
        result_tensor.shape().GetNumElements());

    const auto* lhs_tensor = args[0];
    const auto* rhs_tensor = args[1];
    AlignedEigenVector<LhsType> lhs(GetRawPointer<LhsType>(*args[0]),
                                    args[0]->shape().GetNumElements());

    AlignedEigenVector<RhsType> rhs(GetRawPointer<RhsType>(*args[1]),
                                    args[1]->shape().GetNumElements());

    if (lhs_tensor->shape() == rhs_tensor->shape()) {
      result.device(*dctx->eigen_gpu_device()) = lhs.binaryExpr(rhs, Functor());
      return Error::success();
    }
    if (lhs_tensor->shape().GetNumElements() == 1) {
      assert(rhs_tensor->shape().GetNumElements() ==
             result_tensor.shape().GetNumElements());
      result.device(*dctx->eigen_gpu_device()) =
          lhs.broadcast(
                 Eigen::array<int32_t, 1>(args[1]->shape().GetNumElements()))
              .binaryExpr(rhs, Functor());
      return Error::success();
    }
    if (rhs_tensor->shape().GetNumElements() == 1) {
      assert(lhs_tensor->shape().GetNumElements() ==
             result_tensor.shape().GetNumElements());
      result.device(*dctx->eigen_gpu_device()) = lhs.binaryExpr(
          rhs.broadcast(
              Eigen::array<int32_t, 1>(args[0]->shape().GetNumElements())),
          Functor());
      return Error::success();
    }
    switch (result_tensor.shape().GetRank()) {
      case 1:
        HandleAsBroadcast<1>(dctx, lhs_tensor, rhs_tensor, &result_tensor);
        break;
      case 2:
        HandleAsBroadcast<2>(dctx, lhs_tensor, rhs_tensor, &result_tensor);
        break;
      case 3:
        HandleAsBroadcast<3>(dctx, lhs_tensor, rhs_tensor, &result_tensor);
        break;
      case 4:
        HandleAsBroadcast<4>(dctx, lhs_tensor, rhs_tensor, &result_tensor);
        break;
      case 5:
        HandleAsBroadcast<5>(dctx, lhs_tensor, rhs_tensor, &result_tensor);
        break;
      default:
        return MakeStringError(
            "unimplemented broadcasting high-dimensional tensors");
    }
    return Error::success();
  }
};

}  // namespace internal

// Calls the given Functor with given arguments.
//
// The exact definition of Functor is based on Eigen's implementation details.
// See internal::FunctorDispatch for how exactly we identify an Eigen functor.
// Note that since functors are implementation details, our identification is
// only an approximation, and need to be reviewed when new cases are introduced.
template <typename FunctorSignature, typename Functor>
Expected<DenseGpuTensor> ComputeOpViaEigen(
    GpuDispatchContext* dctx, const TensorMetadata& result_md,
    ArrayRef<const DenseGpuTensor*> args) {
  size_t size_in_bytes =
      result_md.dtype.GetHostSize() * result_md.shape.GetNumElements();

  TFRT_ASSIGN_OR_RETURN(RCReference<gpu::GpuBuffer> result_buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  gpu::DenseGpuTensor result_tensor(result_md.shape, result_md.dtype,
                                    std::move(result_buffer));

  // Dispatch on the Functor arity.
  auto error =
      internal::FunctorDispatch<FunctorSignature, Functor>::EvaluateFunctor(
          dctx, result_tensor, args);

  if (error) {
    return std::move(error);
  }

  return result_tensor;
}

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_OPS_TF_EIGEN_HELPER_CU_H_
