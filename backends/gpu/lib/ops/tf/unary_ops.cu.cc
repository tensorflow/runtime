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

// Collates list of all unary TF operations.

#include "eigen_helper.cu.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"

namespace tfrt {
namespace gpu {
namespace {

// TODO(timshen): add Eigen::internal::functor_traits for performance.
template <typename T>
struct ReluFunctor
    : Eigen::internal::bind2nd_op<Eigen::internal::scalar_max_op<T>> {
  using Base = Eigen::internal::bind2nd_op<Eigen::internal::scalar_max_op<T>>;
  ReluFunctor() : Base(T(0)) {}
};

template <typename FloatType>
struct FloatTruncateTraits;

template <>
struct FloatTruncateTraits<float> {
  using UIntType = uint32_t;
  static constexpr int num_mantissa_bits = 23;
};

template <>
struct FloatTruncateTraits<double> {
  using UIntType = uint64_t;
  static constexpr int num_mantissa_bits = 52;
};

template <>
struct FloatTruncateTraits<Eigen::half> {
  using UIntType = uint16_t;
  static constexpr int num_mantissa_bits = 10;
};

// TODO(timshen): add Eigen::internal::functor_traits for performance.
template <typename ResultType, typename InputType>
struct CastTruncateFunctor {
  __host__ __device__ InputType Truncate(InputType value) const {
    using InputTypeTraits = FloatTruncateTraits<InputType>;
    using ResultTypeTraits = FloatTruncateTraits<ResultType>;
    using InputIntType = typename InputTypeTraits::UIntType;
    InputIntType int_form;
    std::memcpy(&int_form, &value, sizeof(value));

    int num_truncated_bits =
        std::max<int>(InputTypeTraits::num_mantissa_bits -
                          ResultTypeTraits::num_mantissa_bits,
                      0);
    // Set the last num_truncated_bits to 0.
    int_form &= ~((InputIntType(1) << num_truncated_bits) - 1);
    std::memcpy(&value, &int_form, sizeof(value));
    return value;
  }

  __host__ __device__ ResultType operator()(InputType value) const {
    return Eigen::internal::scalar_cast_op<InputType, ResultType>()(
        Truncate(value));
  }

  // TODO(timshen): add packetOp for performance.
};

// ComputeUnaryElementwiseOpViaEigen<Functor, K0, K1, K2> is a TFRT kernel for
// an operation implemented by the Eigen functor `Functor` and supports DType
// kinds K0, K1, K2 (via dynamic dispatch).

template <template <typename> class Functor>
Expected<gpu::DenseGpuTensor> ComputeUnaryElementwiseOpViaEigen(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& input,
    const TensorMetadata& result_md) {
  return MakeStringError("unexpected type for operation");
}

template <template <typename> class Functor, DType::Kind CurrentKind,
          DType::Kind... TrailingKinds>
Expected<gpu::DenseGpuTensor> ComputeUnaryElementwiseOpViaEigen(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& input,
    const TensorMetadata& result_md) {
  if (CurrentKind == result_md.dtype.kind()) {
    using T = EigenTypeForDTypeKind<CurrentKind>;
    using SpecializedFunctor = Functor<T>;
    return gpu::ComputeOpViaEigen<gpu::FunctorSignature<T, T>,
                                  SpecializedFunctor>(dctx, result_md,
                                                      {&input});
  } else {
    return ComputeUnaryElementwiseOpViaEigen<Functor, TrailingKinds...>(
        dctx, input, result_md);
  }
}

template <typename... Args>
struct CastImpl;

template <>
struct CastImpl<> {
  static Expected<gpu::DenseGpuTensor> Invoke(GpuDispatchContext* dctx,
                                              const gpu::DenseGpuTensor& input,
                                              const OpAttrsRef& attrs,
                                              const TensorMetadata& result_md) {
    return MakeStringError("unexpected type for CastOp");
  }
};

template <typename ResultType, typename InputType, typename... Rest>
struct CastImpl<gpu::FunctorSignature<ResultType, InputType>, Rest...> {
  static Expected<gpu::DenseGpuTensor> Invoke(GpuDispatchContext* dctx,
                                              const gpu::DenseGpuTensor& input,
                                              const OpAttrsRef& attrs,
                                              const TensorMetadata& result_md) {
    if (GetDType<ResultType>() == result_md.dtype &&
        GetDType<InputType>() == input.dtype()) {
      bool truncate;
      if (attrs.Get("Truncate", &truncate) && truncate) {
        return gpu::ComputeOpViaEigen<
            gpu::FunctorSignature<ResultType, InputType>,
            CastTruncateFunctor<ResultType, InputType>>(dctx, result_md,
                                                        {&input});
      }
      return gpu::ComputeOpViaEigen<
          gpu::FunctorSignature<ResultType, InputType>,
          Eigen::internal::scalar_cast_op<InputType, ResultType>>(
          dctx, result_md, {&input});
    }
    return CastImpl<Rest...>::Invoke(dctx, input, attrs, result_md);
  }
};

}  // namespace

void RegisterUnaryGpuTfOps(GpuOpRegistry* registry) {
  registry->AddOp("tf.Tanh", TFRT_GPU_OP(ComputeUnaryElementwiseOpViaEigen<
                                         Eigen::internal::scalar_tanh_op,
                                         DType::F16, DType::F32, DType::F64>));
  registry->AddOp(
      "tf.Cast",
      TFRT_GPU_OP(
          CastImpl<gpu::FunctorSignature<float, float>,
                   gpu::FunctorSignature<float, double>,
                   gpu::FunctorSignature<float, Eigen::half>,
                   gpu::FunctorSignature<double, float>,
                   gpu::FunctorSignature<double, double>,
                   gpu::FunctorSignature<double, Eigen::half>,
                   gpu::FunctorSignature<Eigen::half, float>,
                   gpu::FunctorSignature<Eigen::half, double>,
                   gpu::FunctorSignature<Eigen::half, Eigen::half>>::Invoke),
      {"Truncate"});
}
}  // namespace gpu
}  // namespace tfrt
