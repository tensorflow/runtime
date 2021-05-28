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

// Unit test helpers for GPU wrapper tests.
#ifndef THIRD_PARTY_TF_RUNTIME_CPP_TESTS_GPU_STREAM_COMMON_H_
#define THIRD_PARTY_TF_RUNTIME_CPP_TESTS_GPU_STREAM_COMMON_H_

#include <ostream>

#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

class Test : public testing::TestWithParam<wrapper::Platform> {};

// Google Test outputs to std::ostream. Provide ADL'able overloads.
template <typename T>
std::ostream& operator<<(std::ostream& os, T item) {
  llvm::raw_os_ostream raw_os(os);
  raw_os << item;
  return os;
}

// Return the current context or die if an error occurs. This is intended for
// passing CurrentContext instances as temporary to simplify test code. Do not
// use unless the code following it requires zero CurrentContext instances.
inline CurrentContext Current() {
  auto current = CtxGetCurrent();
  cantFail(current.takeError());
  return *current;
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // THIRD_PARTY_TF_RUNTIME_CPP_TESTS_GPU_STREAM_COMMON_H_
