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

// This file implements random number generator.

#include "tfrt/support/random_util.h"

#include <mutex>
#include <random>

namespace tfrt {
namespace random {

namespace {
std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}
}  // namespace

uint64_t New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  static std::mutex mu;
  std::lock_guard<std::mutex> l(mu);
  return (*rng)();
}

}  // namespace random
}  // namespace tfrt
