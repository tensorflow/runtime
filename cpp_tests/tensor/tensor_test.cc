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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

using testing::ElementsAre;

TEST(TensorTest, ScalarDenseHostTensorView) {
  auto host_buffer = HostBuffer::CreateFromExternal(
      /*ptr=*/nullptr, /*size=*/0, [](void*, size_t) {});
  std::array<Index, 1> dims{0};
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

TEST(TensorTest, CreateScalar) {
  auto context = CreateHostContext();
  auto tensor = DenseHostTensor::CreateScalar(1, context.get()).getValue();
  EXPECT_EQ(*tensor.data<int>(), 1);
}

TEST(TensorTest, CompareTensors) {
  auto context = CreateHostContext();
  auto a = DenseHostTensor::CreateScalar(1, context.get()).getValue();
  auto b = DenseHostTensor::CreateScalar(1, context.get()).getValue();
  auto c = DenseHostTensor::CreateScalar(2, context.get()).getValue();
  auto d = DenseHostTensor::CreateScalar(2.5f, context.get()).getValue();
  // This requires that DHT implements operator<< for std::ostream.
  EXPECT_EQ(a, a.CopyRef());
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
}

TEST(TensorTest, CreateMetadataAndComparison) {
  auto a = TensorMetadata::Create<float>(1, 3);
  auto b = TensorMetadata::Create<float>(1, 3);
  auto c = TensorMetadata::Create<int>(3);
  // This requires that `TensorMetadata` implements operator<< for std::ostream.
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(TensorTest, ChipDenseHostTensor) {
  auto context = CreateHostContext();
  auto dht = DenseHostTensor::CreateUninitialized(
                 TensorMetadata::Create<int>(3, 2), context.get())
                 .getValue();
  MutableDHTArrayView<int> view(&dht);
  for (int i = 0; i < view.NumElements(); i++) {
    view[i] = i;
  }
  EXPECT_THAT(view.Elements(), ElementsAre(0, 1, 2, 3, 4, 5));

  auto first_slice = Chip(dht, {1});
  MutableDHTIndexableView<int, 1> first_view(&first_slice);
  first_view.ElementAt(0) = 12;
  first_view.ElementAt(1) = 13;
  EXPECT_THAT(view.Elements(), ElementsAre(0, 1, 12, 13, 4, 5));

  auto second_slice = Chip(dht, {1, 1});
  MutableDHTIndexableView<int, 0> second_view(&second_slice);
  second_view.ElementAt() = 23;
  EXPECT_THAT(view.Elements(), ElementsAre(0, 1, 12, 23, 4, 5));
}

TEST(TensorTest, CopyTo) {
  auto context = CreateHostContext();
  auto a = DenseHostTensor::CreateScalar(1, context.get()).getValue();
  auto b = DenseHostTensor::CreateScalar(2, context.get()).getValue();
  EXPECT_TRUE(CopyTo(a, &b));
  EXPECT_EQ(a, b);
  auto c = DenseHostTensor::CreateScalar(true, context.get()).getValue();
  EXPECT_FALSE(CopyTo(a, &c));
}

TEST(TensorTest, FlattenScalar) {
  auto context = CreateHostContext();
  auto dht = DenseHostTensor::CreateScalar(1, context.get()).getValue();
  EXPECT_EQ(dht.shape().GetRank(), 0);
  auto flat = Flatten(dht);
  EXPECT_EQ(flat.shape().GetRank(), 1);
}

TEST(TensorTest, FlattenTensor) {
  auto context = CreateHostContext();
  auto dht = DenseHostTensor::CreateUninitialized(
                 TensorMetadata::Create<int>(3, 2), context.get())
                 .getValue();
  auto flat = Flatten(dht);
  EXPECT_EQ(flat.shape().GetRank(), 1);
  EXPECT_EQ(flat.shape().GetDimensionSize(0), 6);
  EXPECT_TRUE(dht.buffer().get() == flat.buffer().get());
}

}  // namespace
}  // namespace tfrt
