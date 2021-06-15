/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// Unit test for tensor_serialize_utils.

#include "tfrt/tensor/tensor_serialize_utils.h"

#include "gtest/gtest.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/tensor/btf.h"
#include "tfrt/tensor/btf_util.h"

namespace tfrt {
namespace {

class TensorSerializeUtilsTest : public ::testing::Test {
 protected:
  TensorSerializeUtilsTest()
      : host_context_(CreateHostContext()),
        dht_(DenseHostTensor::CreateUninitialized<float>(TensorShape({2, 2}),
                                                         host_context_.get())
                 .getValue()) {
    MutableDHTArrayView<float> tensor_view(&dht_);
    tensor_view[0] = 1.0f;
    tensor_view[1] = 2.0f;
    tensor_view[2] = 3.0f;
    tensor_view[3] = 4.0f;
  }

  std::unique_ptr<HostContext> host_context_;
  DenseHostTensor dht_;
};

TEST_F(TensorSerializeUtilsTest, SerializeDenseHostTensorToDenseAttr) {
  BefAttrEncoder encoder;
  const size_t offset = SerializeDenseHostTensorToDenseAttr(dht_, &encoder);
  auto dense_attr_buffer = encoder.TakeResult();
  DenseAttr dense_attr(dense_attr_buffer.data() + offset);

  EXPECT_EQ(dense_attr.dtype(), DType::F32);
  EXPECT_EQ(dense_attr.GetElement<float>(0), 1.0f);
  EXPECT_EQ(dense_attr.GetElement<float>(1), 2.0f);
  EXPECT_EQ(dense_attr.GetElement<float>(2), 3.0f);
  EXPECT_EQ(dense_attr.GetElement<float>(3), 4.0f);
}

TEST_F(TensorSerializeUtilsTest, DeserializeDenseHostTensorFromDenseAttr) {
  BefAttrEncoder encoder;
  const size_t offset = SerializeDenseHostTensorToDenseAttr(dht_, &encoder);
  auto dense_attr_buffer = encoder.TakeResult();
  DenseAttr dense_attr(dense_attr_buffer.data() + offset);

  auto dht_des =
      DeserializeDenseHostTensorFromDenseAttr(dense_attr, host_context_.get());
  EXPECT_TRUE(!!dht_des);

  DenseHostTensor dht1(std::move(dht_des.get()));
  MutableDHTArrayView<float> tensor_view1(&dht1);
  EXPECT_EQ(tensor_view1[0], 1.0f);
  EXPECT_EQ(tensor_view1[1], 2.0f);
  EXPECT_EQ(tensor_view1[2], 3.0f);
  EXPECT_EQ(tensor_view1[3], 4.0f);
}

TEST_F(TensorSerializeUtilsTest, SerializeDeserialzeTensorMetadata) {
  auto serialized = SerializeTensorMetadata(dht_.metadata());
  auto expected = DeserializeTensorMetadata(serialized);

  auto md = expected.get();
  EXPECT_EQ(md.dtype.kind(), DType::F32);
  EXPECT_EQ(md.shape.GetRank(), 2);
  EXPECT_EQ(md.shape.GetDimensionSize(0), 2);
  EXPECT_EQ(md.shape.GetDimensionSize(1), 2);
}

TEST_F(TensorSerializeUtilsTest, SerializeDeserialzeDenseHostTensor) {
  auto serialized = SerializeDenseHostTensor(dht_, host_context_.get());
  auto expected =
      DeserializeDenseHostTensor(serialized.get(), host_context_.get());
  EXPECT_EQ(expected.get(), dht_);
}

}  // namespace
}  // namespace tfrt
