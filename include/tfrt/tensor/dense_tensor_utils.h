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

// This file defines a few helper methods for DenseHostTensor.
#ifndef TFRT_TENSOR_DENSE_TENSOR_UTILS_H_
#define TFRT_TENSOR_DENSE_TENSOR_UTILS_H_

#include <complex>
#include <type_traits>

#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {

// Compares two tenors using the provided function.
template <typename T, typename F>
bool TensorEqual(const DenseHostTensor& lhs, const DenseHostTensor& rhs,
                 F&& cmp) {
  if (lhs.metadata() != rhs.metadata()) return false;
  DHTArrayView<T> lhs_view(&lhs);
  DHTArrayView<T> rhs_view(&rhs);
  assert(lhs_view.NumElements() == rhs_view.NumElements());
  return std::equal(lhs_view.begin(), lhs_view.end(), rhs_view.begin(), cmp);
}

// Compare floating point numbers for equality within a given ULP (units in the
// last place).
template <typename T, int ULP,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
bool TensorElementsClose(T x, T y) {
  // The machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs.
  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ULP
         // Unless the result is subnormal.
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

// Comparing complex numbers for equality within a given ULP.
template <typename T, int ULP,
          std::enable_if_t<is_complex_t<T>::value, int> = 0>
bool TensorElementsClose(T x, T y) {
  return TensorElementsClose<typename T::value_type, ULP>(x.real(), y.real()) &&
         TensorElementsClose<typename T::value_type, ULP>(x.imag(), y.imag());
}

// Comparing integer types is equivalent to equality.
template <typename T, int ULP = 0 /* Unused */,
          std::enable_if_t<std::is_integral<T>::value, int> = 0>
bool TensorElementsClose(T x, T y) {
  return x == y;
}

template <typename T, int ULP = 2>
bool TensorApproxEqual(const DenseHostTensor& lhs, const DenseHostTensor& rhs) {
  return TensorEqual<T>(lhs, rhs, TensorElementsClose<T, ULP>);
}

template <typename T>
bool TensorEqual(const DenseHostTensor& lhs, const DenseHostTensor& rhs) {
  return TensorEqual<T>(lhs, rhs, [](T x, T y) { return x == y; });
}

// Chip is a special kind of slice. It indexes into the view at the given
// coordinate prefix and returns a view onto the remaining dimensions.
// It is similar to indexing into a numpy array, e.g. for a 5D ndarray A, the
// slice A[1, 3] would return a 3D view.
DenseHostTensor Chip(const DenseHostTensor& tensor, ArrayRef<ssize_t> dims);

// Flattens a DHT into a Rank-1 DHT that will share the underlying HostBuffer.
DenseHostTensor Flatten(const DenseHostTensor& tensor);

// Copies the data of a DHT to another DHT. Returns false if the metadatas of
// the DHTs do not match.
LLVM_NODISCARD bool CopyTo(const DenseHostTensor& src, DenseHostTensor* dst);

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_TENSOR_UTILS_H_
