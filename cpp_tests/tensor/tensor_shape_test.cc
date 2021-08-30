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

// Unit test for TFRT TensorShape.

#include "tfrt/tensor/tensor_shape.h"

#include <array>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(TensorShapeTest, Strides) {
  std::array<Index, 3> dims = {16, 15, 14};
  TensorShape shape(dims);

  std::array<Index, 3> strides;
  shape.GetStrides(&strides);

  std::array<Index, 3> expected = {15 * 14, 14, 1};
  EXPECT_EQ(strides, expected);
}

}  // namespace
}  // namespace tfrt
