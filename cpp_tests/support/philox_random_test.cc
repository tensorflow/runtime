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

// Tests related to the philox random number generator.

#include "tfrt/support/philox_random.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(PhiloxRandomTest, GenerateNumber) {
  random::PhiloxRandom generator(100, 200);
  EXPECT_THAT(generator(), 3747298259);
  EXPECT_THAT(generator(), 3724722508);
}

}  // namespace
}  // namespace tfrt
