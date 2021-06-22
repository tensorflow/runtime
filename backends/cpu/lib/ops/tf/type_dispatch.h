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

// ---=====
//
// Helper functions to do Tf ops type based dispatching.

#ifndef TFRT_BACKENDS_CPU_OPS_TF_DISPATCH_OP_H_
#define TFRT_BACKENDS_CPU_OPS_TF_DISPATCH_OP_H_

#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {
namespace internal {

// Type based dispatching for Tensorflow ops.
//
// Example: Sqrt for float or int32_t tensors. Sqrt is not defined for bool.
//
//   DenseHostTensor tensor = ...;
//   TypeDispatch<float, int32_t> type_dispatch(tensor.dtype());
//
//   // This lambda will be called if tensor dtype is not supported.
//   auto unsupported = [](Dtype dtype) -> Error {
//     return MakeStringError("Unsupported dtype: ", dtype);
//   };
//
//   // This template lambda will be instantiated only for `float` and
//   // `int32_t`, and will be called if `tensor.dtype()` is one of them.
//   auto sqrt = [&](auto type_tag) -> Error {
//     using T = decltype(type_tag);
//
//     // This expression will not compile for `bool`.
//     auto eigen_t = AsEigenTensor<T>(tensor);
//     eigen_t = eigen_t.sqrt();
//
//     return Error::success();
//   };
//
//   Error result = type_dispatch(sqrt, unsupported);
//
template <typename... Types>
class TypeDispatch {
 public:
  explicit TypeDispatch(DType dtype) : dtype_(dtype) {}

  template <typename Dispatch, typename Unsupported>
  auto operator()(Dispatch&& dispatch, Unsupported&& unsupported) {
    return Impl<Dispatch, Unsupported, Types...>(
        std::forward<Dispatch>(dispatch),
        std::forward<Unsupported>(unsupported));
  }

 private:
  // Variadic template recursion that compares first type of the variadic pack
  // with `dtype`. If type matches it calls `dispatch` with a type tag (default
  // constructed value of Eigen compatibe type derived from T), otherwise it
  // will try to match the next type in the variadic pack.
  template <typename Dispatch, typename Unsupported, typename T,
            typename... Rest>
  auto Impl(Dispatch&& dispatch, Unsupported&& unsupported) {
    if (dtype_ == GetDType<T>()) {
      return dispatch(EigenTypeForDTypeKind<GetDType<T>()>{});
    }

    return Impl<Dispatch, Unsupported, Rest...>(
        std::forward<Dispatch>(dispatch),
        std::forward<Unsupported>(unsupported));
  }

  // Variadic template recursion base case, pass unsupported dtype to
  // `unsupported` callback.
  template <typename Dispatch, typename Unsupported>
  auto Impl(Dispatch&& dispatch, Unsupported&& unsupported) {
    return unsupported(dtype_);
  }

  const DType dtype_;
};

template <DType... kind>
struct GetTypeDispatch {
  using Type = internal::TypeDispatch<TypeForDTypeKind<kind>...>;
};

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_OPS_TF_DISPATCH_OP_H_
