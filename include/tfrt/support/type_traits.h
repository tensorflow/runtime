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

//===- type_traits.h --------------------------------------------*- C++ -*-===//
//
// This file defines type traits related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_SUPPORT_TYPE_TRAITS_H_
#define TFRT_SUPPORT_TYPE_TRAITS_H_

#include <utility>

namespace tfrt {

// Utility template for tag dispatching.
template <typename T>
struct TypeTag {};

// This is the equivalent of std::void_t in C++17.
template <typename... Ts>
struct make_void {
  typedef void type;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// This is the equivalent of std::conjunction in C++17.
template <class...>
struct conjunction : std::true_type {};
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

// This is the equivalent of std::negation in C++17.
template <class B>
struct negation : std::integral_constant<bool, !bool(B::value)> {};

// Check whether T may be a base class.
template <typename T>
using MaybeBase = conjunction<std::is_class<T>, negation<std::is_final<T>>>;

// The detector pattern in C++ that can be used for checking whether a type has
// a specific property, e.g. whether an internal type is present or whether a
// particular operation is valid.
//
// Sample usage:
//
// struct Foo {
//   using difference_type = int;
//   int get();
// };
// struct Bar {};
//
// // Check whether a type T has an internal difference_type.
// template<class T>
// using diff_t = typename T::difference_type;
//
// static_assert(is_detected_v<diff_t, Foo>, "Foo has difference_type");
// static_assert(!is_detected_v<diff_t, Bar>, "Bar has no difference_type");
//
// // Check whether a type T has a get() member function.
// template<class T>
// using has_get_t = decltype(std::declval<T>().get());
//
// static_assert(is_detected_v<has_get_t, Foo>, "Foo has get()");
// static_assert(!is_detected_v<has_get_t, Bar>, "Bar has no get()");
//
// See https://en.cppreference.com/w/cpp/experimental/is_detected for details.

namespace internal {

// nonesuch is a class type used to indicate detection failure.
struct nonesuch {
  ~nonesuch() = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

template <class Default, class AlwaysVoid, template <class...> class Op,
          class... Args>
struct detector : std::false_type {
  using value_t = std::false_type;
  using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type = Op<Args...>;
};

}  // namespace internal

template <template <class...> class Op, class... Args>
using is_detected =
    typename internal::detector<internal::nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t =
    typename internal::detector<internal::nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = internal::detector<Default, void, Op, Args...>;

template <template <class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_TYPE_TRAITS_H_
