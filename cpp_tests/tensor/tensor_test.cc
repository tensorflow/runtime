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

// Unit test for TFRT Tensor.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

using testing::ElementsAre;

TEST(TensorTest, ScalarDenseHostTensorView) {
  auto host_buffer = HostBuffer::CreateFromExternal(
      /*ptr=*/nullptr, /*size=*/0, [](void*, size_t) {});
  std::array<ssize_t, 1> dims{0};
  DenseHostTensor tensor(TensorMetadata(DType(DType::I32), TensorShape(dims)),
                         std::move(host_buffer));
  DHTArrayView<int32_t> view(&tensor);
  EXPECT_TRUE(view.Elements().empty());
}

TEST(TensorTest, ChipDenseHostTensorView) {
  std::array<int32_t, 6> data{0, 1, 2, 3, 4, 5};

  DHTIndexableView<int32_t, 2> view(data.data(), 3, 2);
  EXPECT_THAT(view.Chip(1).FixedShape(), ElementsAre(2));
  EXPECT_THAT(view.Chip(1).Elements(), ElementsAre(2, 3));
  EXPECT_THAT(view.Chip(1, 1).FixedShape(), ElementsAre());
  EXPECT_THAT(view.Chip(1, 1).Elements(), ElementsAre(3));

  MutableDHTIndexableView<int32_t, 2> mut_view(data.data(), 3, 2);
  mut_view.Chip(1).ElementAt(0) = 4;
  EXPECT_THAT(mut_view.Chip(1).FixedShape(), ElementsAre(2));
  EXPECT_THAT(mut_view.Chip(1).Elements(), ElementsAre(4, 3));
  EXPECT_THAT(mut_view.Chip(1, 1).FixedShape(), ElementsAre());
  EXPECT_THAT(mut_view.Chip(1, 1).Elements(), ElementsAre(3));
}

TEST(TensorTest, MutableIndexableViewCanBeCastedToIndexableView) {
  auto fn = [](const DHTIndexableView<int32_t, 2>& view) {
    return view.ElementAt(1, 0);
  };
  std::array<int32_t, 6> data{0, 1, 2, 3, 4, 5};
  MutableDHTIndexableView<int32_t, 2> mut_view(data.data(), 3, 2);
  EXPECT_EQ(fn(mut_view), 2);
}

}  // namespace
}  // namespace tfrt
