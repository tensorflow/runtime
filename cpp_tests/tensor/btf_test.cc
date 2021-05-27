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

// Unit test for BTF utils.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/tensor/btf_util.h"

namespace tfrt {
namespace {

TEST(BTFTest, BTFWriteAndRead) {
  auto context = CreateHostContext();
  const auto a = CreateDummyTensor<int>({3, 2}, context.get());
  const auto b = CreateDummyTensor<uint8_t>({63}, context.get());
  const auto c = CreateDummyTensor<uint64_t>({}, context.get());
  std::vector<const Tensor*> tensors{&a, &b, &c};
  std::stringstream os;
  EXPECT_FALSE(WriteTensorsToBTF(&os, tensors));
  const std::string buffer = os.str();
  std::istringstream is(buffer);
  auto offsets = ReadBTFOffsets(&is).get();
  EXPECT_EQ(offsets.size(), tensors.size());
  EXPECT_EQ(offsets[0], 32);
  for (int i = 0; i < tensors.size(); i++) {
    const auto& expected =
        reinterpret_cast<const DenseHostTensor&>(*tensors[i]);
    auto out = ReadDHTFromBTF(&is, offsets[i], context.get());
    EXPECT_EQ(*out, expected);
  }
}

}  // namespace
}  // namespace tfrt
