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

#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {

// Compares two tenors using the provided function.
template <typename T, typename F>
bool CompareTensors(const DHTArrayView<T> lhs, const DHTArrayView<T> rhs,
                    F&& cmp) {
  if (lhs.Shape() == rhs.Shape()) {
    auto lelements = lhs.Elements();
    auto relements = rhs.Elements();
    assert(lelements.size() == relements.size());
    return std::equal(lelements.begin(), lelements.end(), relements.begin(),
                      cmp);
  }
  return false;
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
bool AllElementsClose(DHTArrayView<T> lhs, DHTArrayView<T> rhs) {
  return CompareTensors<T>(lhs, rhs, TensorElementsClose<T, ULP>);
}

template <typename T>
bool operator==(const DHTArrayView<T> lhs, const DHTArrayView<T> rhs) {
  return CompareTensors<T>(lhs, rhs, [](T x, T y) { return x == y; });
}

template <typename T>
bool operator!=(const DHTArrayView<T> lhs, const DHTArrayView<T> rhs) {
  return !(lhs == rhs);
}

}  // namespace tfrt

#endif  // TFRT_TENSOR_DENSE_TENSOR_UTILS_H_
