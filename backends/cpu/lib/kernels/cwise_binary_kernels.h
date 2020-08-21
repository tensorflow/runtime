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

//===- cwise_binary_kernels.h -----------------------------------*- C++ -*-===//
//
// Coefficient wise binary kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_BINARY_KERNELS_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_BINARY_KERNELS_H_

#include "tfrt/common/compat/eigen/eigen_evaluator.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/common/ops/tf/bcast.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/scalar_host_tensor.h"

namespace tfrt {
namespace cpu {
namespace functor {

// Forward declare unary functors for binding scalar arguments
template <typename T, typename R, typename Functor>
struct BindRightScalar;
template <typename T, typename R, typename Functor>
struct BindLeftScalar;

}  // namespace functor
}  // namespace cpu
}  // namespace tfrt

namespace Eigen {
namespace internal {

// Register unary functors for binding scalar arguments with Eigen.
template <typename T, typename R, typename Functor>
struct functor_traits<tfrt::cpu::functor::BindRightScalar<T, R, Functor>> {
  enum {
    Cost = functor_traits<Functor>::Cost,
    PacketAccess = functor_traits<Functor>::PacketAccess,
  };
};

template <typename T, typename R, typename Functor>
struct functor_traits<tfrt::cpu::functor::BindLeftScalar<T, R, Functor>> {
  enum {
    Cost = functor_traits<Functor>::Cost,
    PacketAccess = functor_traits<Functor>::PacketAccess,
  };
};

}  // namespace internal
}  // namespace Eigen

namespace tfrt {
namespace cpu {
namespace functor {

template <typename T, typename F, typename R = T>
struct BinaryFunctor {
  using Functor = F;
  using Input = T;
  using Output = R;
};

struct Add {
  template <typename T>
  using Functor = BinaryFunctor<T, Eigen::internal::scalar_sum_op<T>>;
};

struct Div {
  template <typename T>
  using Functor = BinaryFunctor<T, Eigen::internal::scalar_quotient_op<T>>;
};

struct Sub {
  template <typename T>
  using Functor = BinaryFunctor<T, Eigen::internal::scalar_difference_op<T>>;
};

struct Mul {
  template <typename T>
  using Functor = BinaryFunctor<T, Eigen::internal::scalar_product_op<T>>;
};

struct Less {
  template <typename T>
  using Functor = BinaryFunctor<
      T, Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>, bool>;
};

// Bind scalar value on the right side of the binary expression to the binary
// functor and get back a unary functor:
//
//  binary_functor(_, value) ->  unary_functor(_).
template <typename T, typename R, typename Functor>
struct BindRightScalar : private Functor {
  using result_type = R;
  using Packet = typename Eigen::internal::packet_traits<T>::type;

  template <typename... Args>
  explicit BindRightScalar(const T& value, Args... args)
      : Functor(args...), right_value(value) {
    right_packet = Eigen::internal::pset1<Packet>(right_value);
  }

  R operator()(const T& left_value) const {
    return Functor::operator()(left_value, right_value);
  }

  Packet packetOp(const Packet& left_packet) const {
    return Functor::packetOp(left_packet, right_packet);
  }

  const T right_value;
  Packet right_packet;
};

// Bind scalar value on the left side of the binary expression to the binary
// functor and get back a unary functor:
//
//  binary_functor(value, _) ->  unary_functor(_).
template <typename T, typename R, typename Functor>
struct BindLeftScalar : private Functor {
  using result_type = R;
  using Packet = typename Eigen::internal::packet_traits<T>::type;

  template <typename... Args>
  explicit BindLeftScalar(const T& value, Args... args)
      : Functor(args...), left_value(value) {
    left_packet = Eigen::internal::pset1<Packet>(left_value);
  }

  R operator()(const T& right_value) const {
    return Functor::operator()(left_value, right_value);
  }

  Packet packetOp(const Packet& right_packet) const {
    return Functor::packetOp(left_packet, right_packet);
  }

  const T left_value;
  Packet left_packet;
};

}  // namespace functor

namespace internal {

template <typename BinaryFunctor, typename EigenEvaluator>
struct BinaryKernelImpl {
  using Functor = typename BinaryFunctor::Functor;
  using Input = typename BinaryFunctor::Input;
  using Output = typename BinaryFunctor::Output;

  explicit BinaryKernelImpl(EigenEvaluator eigen_evaluator)
      : eigen{eigen_evaluator} {}

  template <typename OnDone>
  void ScalarScalar(const HostTensor& lhs, const HostTensor& rhs,
                    HostTensor* output, OnDone on_done) {
    auto* lhs_scalar = cast<ScalarHostTensor<Input>>(&lhs);
    auto* rhs_scalar = cast<ScalarHostTensor<Input>>(&rhs);
    auto* out_scalar = cast<ScalarHostTensor<Output>>(output);

    Functor functor;
    Output value = functor(lhs_scalar->GetValue(), rhs_scalar->GetValue());
    out_scalar->SetValue(value);

    on_done(Error::success());
  }

  template <typename OnDone>
  void ScalarTensor(const HostTensor& lhs, const HostTensor& rhs,
                    HostTensor* output, OnDone on_done) {
    auto* lhs_scalar = cast<ScalarHostTensor<Input>>(&lhs);
    auto* rhs_tensor = cast<DenseHostTensor>(&rhs);
    auto* out_tensor = cast<DenseHostTensor>(output);

    auto rhs_t = compat::AsEigenConstTensor(DHTArrayView<Input>(rhs_tensor));
    auto out_t = compat::AsEigenTensor(MutableDHTArrayView<Output>(out_tensor));

    // Bind scalar value to the right side of the binary functor.
    using BindLeft = functor::BindLeftScalar<Input, Output, Functor>;
    auto expr = rhs_t.unaryExpr(BindLeft(lhs_scalar->GetValue()));

    eigen.Evaluate(
        out_t, std::move(expr),
        [buffers = eigen.KeepAlive(rhs_tensor, out_tensor),
         on_done = std::move(on_done)]() { on_done(Error::success()); });
  }

  template <typename OnDone>
  void TensorScalar(const HostTensor& lhs, const HostTensor& rhs,
                    HostTensor* output, OnDone on_done) {
    auto* lhs_tensor = cast<DenseHostTensor>(&lhs);
    auto* rhs_scalar = cast<ScalarHostTensor<Input>>(&rhs);
    auto* out_tensor = cast<DenseHostTensor>(output);

    auto lhs_t = compat::AsEigenConstTensor(DHTArrayView<Input>(lhs_tensor));
    auto out_t = compat::AsEigenTensor(MutableDHTArrayView<Output>(out_tensor));

    // Bind scalar value to the right side of the binary functor.
    using BindRight = functor::BindRightScalar<Input, Output, Functor>;
    auto expr = lhs_t.unaryExpr(BindRight(rhs_scalar->GetValue()));

    eigen.Evaluate(
        out_t, std::move(expr),
        [buffers = eigen.KeepAlive(lhs_tensor, out_tensor),
         on_done = std::move(on_done)]() { on_done(Error::success()); });
  }

  template <typename OnDone>
  void TensorTensor(const HostTensor& lhs, const HostTensor& rhs,
                    HostTensor* output, OnDone on_done) {
    auto* lhs_tensor = cast<DenseHostTensor>(&lhs);
    auto* rhs_tensor = cast<DenseHostTensor>(&rhs);
    auto* out_tensor = cast<DenseHostTensor>(output);

    auto lhs_t = compat::AsEigenConstTensor(DHTArrayView<Input>(lhs_tensor));
    auto rhs_t = compat::AsEigenConstTensor(DHTArrayView<Input>(rhs_tensor));
    auto out_t = compat::AsEigenTensor(MutableDHTArrayView<Output>(out_tensor));

    // Builds a callback for assign operations that extends buffers lifetime.
    auto assign_callback = [&]() {
      return [buffers = eigen.KeepAlive(lhs_tensor, rhs_tensor, out_tensor),
              on_done = std::move(on_done)]() { on_done(Error::success()); };
    };

    if (lhs_tensor->shape() == rhs_tensor->shape()) {
      // Arguments do not need broadcasting.
      auto expr = lhs_t.binaryExpr(rhs_t, Functor());
      eigen.Evaluate(out_t, std::move(expr), assign_callback());

    } else if (lhs_tensor->NumElements() == 1) {
      // Scalar (or Tensor of size 1) + Tensor.
      using BindLeft = functor::BindLeftScalar<Input, Output, Functor>;
      auto expr = rhs_t.unaryExpr(BindLeft(*lhs_t.data()));
      eigen.Evaluate(out_t, std::move(expr), assign_callback());

    } else if (rhs_tensor->NumElements() == 1) {
      // Tensor + Scalar (or Tensor of size 1).
      using BindRight = functor::BindRightScalar<Input, Output, Functor>;
      auto expr = lhs_t.unaryExpr(BindRight(*rhs_t.data()));
      eigen.Evaluate(out_t, std::move(expr), assign_callback());

    } else {
      // Handle Tensor + Tensor with broadcasting.
      TensorTensorBcast(*lhs_tensor, *rhs_tensor, out_tensor,
                        std::move(on_done));
    }
  }

  template <typename OnDone>
  void TensorTensorBcast(const DenseHostTensor& lhs_tensor,
                         const DenseHostTensor& rhs_tensor,
                         DenseHostTensor* out_tensor, OnDone on_done) {
    // Get broadcasting specifications for lhs and rhs arguments.
    auto lhs_bcast = GetArgumentBCast(lhs_tensor.shape(), out_tensor->shape());
    auto rhs_bcast = GetArgumentBCast(rhs_tensor.shape(), out_tensor->shape());

    if (auto err = lhs_bcast.takeError()) {
      on_done(std::move(err));
      return;
    }
    if (auto err = rhs_bcast.takeError()) {
      on_done(std::move(err));
      return;
    }

    const int rank = lhs_bcast->rank();
    assert(lhs_bcast->rank() == rhs_bcast->rank());

    auto lhs_arr_view = DHTArrayView<Input>(&lhs_tensor);
    auto rhs_arr_view = DHTArrayView<Input>(&rhs_tensor);

    // Use Rank type defined below to pass rank value via lambda argument.
    auto dispatch = [&](auto rank_dispatch) -> void {
      constexpr int rank = decltype(rank_dispatch)::value;

      FixedRankShape<rank> lhs_shape(TensorShape(lhs_bcast->reshape()));
      FixedRankShape<rank> rhs_shape(TensorShape(rhs_bcast->reshape()));

      auto out_view = MutableDHTIndexableView<Output, rank>(out_tensor);
      auto lhs_t = compat::AsEigenConstTensor(lhs_arr_view, lhs_shape);
      auto rhs_t = compat::AsEigenConstTensor(rhs_arr_view, rhs_shape);
      auto out_t = compat::AsEigenTensor(out_view);

      auto lhs_expr = lhs_t.broadcast(lhs_bcast->broadcast());
      auto rhs_expr = rhs_t.broadcast(rhs_bcast->broadcast());
      auto expr = lhs_expr.binaryExpr(rhs_expr, Functor());

      eigen.Evaluate(
          out_t, std::move(expr),
          [buffers = eigen.KeepAlive(&lhs_tensor, &rhs_tensor, out_tensor),
           on_done = std::move(on_done)]() { on_done(Error::success()); });
    };

    // Rank 1 (vectors) must have the same number of elements, or one of the
    // arguments must be a vector of size [1] to be broadcastable. These
    // combinations are handled by TensorTensor function above, and we need to
    // dispatch only tensors with rank 2 or higher.
    if (rank == 2) {
      dispatch(Rank<2>{});
    } else if (rank == 3) {
      dispatch(Rank<3>{});
    } else if (rank == 4) {
      dispatch(Rank<4>{});
    } else if (rank == 5) {
      dispatch(Rank<5>{});
    } else {
      on_done(MakeStringError(
          "Unsupported binary kernel broadcasting: lhs=", lhs_tensor.shape(),
          " rhs=", rhs_tensor.shape(), " out=", out_tensor->shape()));
    }
  }

  // Helper struct to pass compile time constant to lambda as a value argument.
  template <int rank>
  struct Rank {
    static constexpr int value = rank;
  };

  EigenEvaluator eigen;
};

}  // namespace internal

template <typename BinaryFunctor, typename EigenEvaluator, typename OnDone>
void BinaryKernel(const HostTensor& lhs, const HostTensor& rhs,
                  HostTensor* output, const ExecutionContext& exec_ctx,
                  OnDone on_done) {
  using T = typename BinaryFunctor::Input;

  internal::BinaryKernelImpl<BinaryFunctor, EigenEvaluator> impl(
      EigenEvaluator{exec_ctx.host()});

  if (isa<ScalarHostTensor<T>>(lhs) && isa<ScalarHostTensor<T>>(rhs)) {
    impl.ScalarScalar(lhs, rhs, output, std::move(on_done));

  } else if (isa<ScalarHostTensor<T>>(lhs) && isa<DenseHostTensor>(rhs)) {
    impl.ScalarTensor(lhs, rhs, output, std::move(on_done));

  } else if (isa<DenseHostTensor>(lhs) && isa<ScalarHostTensor<T>>(rhs)) {
    impl.TensorScalar(lhs, rhs, output, std::move(on_done));

  } else if (isa<DenseHostTensor>(lhs) && isa<DenseHostTensor>(rhs)) {
    impl.TensorTensor(lhs, rhs, output, std::move(on_done));

  } else {
    on_done(MakeStringError("Unsupported operand types"));
  }
}

template <typename BinaryFunctor>
AsyncValueRef<Chain> BinaryKernel(const HostTensor& lhs, const HostTensor& rhs,
                                  HostTensor* output,
                                  const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  AsyncValueRef<Chain> chain = MakeConstructedAsyncValueRef<Chain>(host);

  auto on_done = [chain = chain.CopyRef()](Error err) {
    err ? chain.SetError(err) : chain.SetStateConcrete();
  };

  BinaryKernel<BinaryFunctor, compat::AsyncEigenEvaluator>(
      lhs, rhs, output, exec_ctx, std::move(on_done));

  return chain;
}

template <typename BinaryFunctor>
Error SyncBinaryKernel(const HostTensor& lhs, const HostTensor& rhs,
                       HostTensor* output, const ExecutionContext& exec_ctx) {
  Error error = Error::success();
  auto on_done = [&](Error err) { error = std::move(err); };
  BinaryKernel<BinaryFunctor, compat::SyncEigenEvaluator>(lhs, rhs, output,
                                                          exec_ctx, on_done);
  return error;
}

}  // namespace cpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_CPU_CWISE_BINARY_KERNELS_H_
