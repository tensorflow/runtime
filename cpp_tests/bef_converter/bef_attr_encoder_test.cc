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

// Unit test for TFRT BefAttrEncoder.

#include "tfrt/bef_converter/bef_attr_encoder.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {

template <typename T>
void TestBasicTypeEncoding(T value) {
  BefAttrEncoder encoder;

  const size_t offset = encoder.EncodeAttr(value);
  auto buffer = encoder.TakeResult();
  Attribute<T> attr(buffer.data() + offset);

  EXPECT_EQ(attr.get(), value);
}

constexpr uint16_t kTestUint8 = 12;
constexpr uint16_t kTestUint16 = 23;
constexpr uint32_t kTestUint32 = 4567;
constexpr uint64_t kTestUint64 = 12345678L;
constexpr float kTestFloat = 3.14;
constexpr float kTestDouble = 3.141592;
TEST(BefAttrEncoderTest, EncodeBasicTypes) {
  TestBasicTypeEncoding(kTestUint8);
  TestBasicTypeEncoding(kTestUint16);
  TestBasicTypeEncoding(kTestUint32);
  TestBasicTypeEncoding(kTestUint64);
  TestBasicTypeEncoding(kTestFloat);
  TestBasicTypeEncoding(kTestDouble);
}

constexpr int32_t kTestInt32Array[] = {1, 2, 3, 4, 5, 6, 7};
constexpr auto kTestInt32ArraySize = sizeof(kTestInt32Array) / sizeof(int32_t);
TEST(BefAttrEncoderTest, EncodeInt32ArrayAttribute) {
  auto input_array_ref =
      llvm::makeArrayRef(kTestInt32Array, kTestInt32ArraySize);

  BefAttrEncoder encoder;

  const size_t offset = encoder.EncodeArrayAttr(input_array_ref);
  auto buffer = encoder.TakeResult();
  ArrayAttribute<int32_t> attr(buffer.data() + offset);
  EXPECT_EQ(kTestInt32ArraySize, attr.size());
  EXPECT_THAT(attr.data(), ::testing::ContainerEq(input_array_ref));
}

TEST(BefAttrEncoderTest, EncodeEmptyArray) {
  BefAttrEncoder encoder;

  const size_t offset =
      encoder.EncodeArrayAttr(llvm::makeArrayRef(kTestInt32Array, 0));
  auto buffer = encoder.TakeResult();
  ArrayAttribute<int32_t> attr(buffer.data() + offset);
  EXPECT_EQ(attr.size(), 0);
}

constexpr double kTestDoubleArray[] = {1.1, 2.2, 3.3, 4.4, 5.5,
                                       6.6, 7.7, 8.8, 9.9};
constexpr auto kTestDoubleArraySize = sizeof(kTestDoubleArray) / sizeof(double);
TEST(BefAttrEncoderTest, EncodeDoubleArrayAttribute) {
  auto input_array_ref =
      llvm::makeArrayRef(kTestDoubleArray, kTestDoubleArraySize);

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeArrayAttr(input_array_ref);
  auto buffer = encoder.TakeResult();
  ArrayAttribute<double> attr(buffer.data() + offset);
  EXPECT_EQ(kTestDoubleArraySize, attr.size());
  EXPECT_THAT(attr.data(), ::testing::ContainerEq(input_array_ref));
}

TEST(BefAttrEncoderTest, EncodeZeroShape) {
  int64_t dims[1];

  BefAttrEncoder encoder;
  const size_t offset =
      encoder.EncodeRankedShapeAttr(llvm::makeArrayRef(dims, 0));

  auto buf = encoder.TakeResult();
  ShapeAttr shape_attr(buf.data() + offset);

  EXPECT_EQ(shape_attr.GetRank(), 0);
  ArrayRef<int64_t> shape = shape_attr.GetShape();
  EXPECT_EQ(shape.size(), 0);
}

TEST(BefAttrEncoderTest, EncodeUnrankedShape) {
  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeUnrankedShapeAttr();

  auto buf = encoder.TakeResult();
  ShapeAttr shape_attr(buf.data() + offset);

  ASSERT_FALSE(shape_attr.HasRank());
}

TEST(BefAttrEncoderTest, EncodeRankedShape) {
  int64_t dims[3] = {1, 2, 3};
  auto input_shape = llvm::makeArrayRef(dims, 3);

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeRankedShapeAttr(input_shape);

  auto buf = encoder.TakeResult();
  ShapeAttr shape_attr(buf.data() + offset);

  EXPECT_EQ(shape_attr.GetRank(), 3);

  EXPECT_THAT(shape_attr.GetShape(), ::testing::ContainerEq(input_shape));
}

TEST(BefAttrEncoderTest, EncodeShapeList) {
  const int64_t a[1] = {1};
  const int64_t b[2] = {2, 3};
  const int64_t c[3] = {4, 5, 6};
  const int64_t* dims[4] = {a, b, c, nullptr};

  int sizes[4] = {1, 2, 3, -1};

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeShapeListAttr(dims, sizes, 4);

  auto buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data() + offset);

  EXPECT_EQ(aggr_attr.GetNumElements(), 4);

  ShapeAttr shape_a = aggr_attr.GetAttributeOfType<ShapeAttr>(0);
  EXPECT_THAT(shape_a.GetShape(),
              ::testing::ContainerEq(llvm::makeArrayRef(a, 1)));

  ShapeAttr shape_b = aggr_attr.GetAttributeOfType<ShapeAttr>(1);
  EXPECT_THAT(shape_b.GetShape(),
              ::testing::ContainerEq(llvm::makeArrayRef(b, 2)));

  ShapeAttr shape_c = aggr_attr.GetAttributeOfType<ShapeAttr>(2);
  EXPECT_THAT(shape_c.GetShape(),
              ::testing::ContainerEq(llvm::makeArrayRef(c, 3)));

  ShapeAttr shape_d = aggr_attr.GetAttributeOfType<ShapeAttr>(3);
  ASSERT_FALSE(shape_d.HasRank());
}

TEST(BefAttrEncoderTest, EncodeEmptyString) {
  BefAttrEncoder encoder;
  std::string empty_string = "";
  const size_t offset =
      encoder.EncodeStringAttr(string_view(empty_string.data(), 0));

  auto buf = encoder.TakeResult();
  StringAttr string_attr(buf.data() + offset);

  EXPECT_EQ(string_attr.GetValue().size(), 0);
}

TEST(BefAttrEncoderTest, EncodeString) {
  BefAttrEncoder encoder;
  std::string sample_string = "tfrt";
  const size_t offset = encoder.EncodeStringAttr(
      string_view(sample_string.data(), sample_string.size()));

  auto buf = encoder.TakeResult();
  StringAttr string_attr(buf.data() + offset);

  string_view sv = string_attr.GetValue();

  EXPECT_EQ(sv.size(), 4);
  EXPECT_EQ(sv, "tfrt");
}

TEST(BefAttrEncoderTest, EncodeStringList) {
  const std::string a = "hi";
  const std::string b = "tfrt";
  const std::string c = "world";

  const void* values[3] = {a.data(), b.data(), c.data()};

  size_t sizes[3] = {a.size(), b.size(), c.size()};

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeStringListAttr(values, sizes, 3);

  auto buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data() + offset);

  EXPECT_EQ(aggr_attr.GetNumElements(), 3);

  StringAttr str_a_attr = aggr_attr.GetAttributeOfType<StringAttr>(0);
  string_view str_a = str_a_attr.GetValue();
  EXPECT_EQ(str_a, a);

  StringAttr str_b_attr = aggr_attr.GetAttributeOfType<StringAttr>(1);
  string_view str_b = str_b_attr.GetValue();
  EXPECT_EQ(str_b, b);

  StringAttr str_c_attr = aggr_attr.GetAttributeOfType<StringAttr>(2);
  string_view str_c = str_c_attr.GetValue();
  EXPECT_EQ(str_c, c);
}

TEST(BefAttrEncoderTest, EncodeFuncList) {
  const std::string a = "tf";
  const std::string b = "new";
  const std::string c = "runtime";

  const void* values[3] = {a.data(), b.data(), c.data()};

  size_t sizes[3] = {a.size(), b.size(), c.size()};

  BefAttrEncoder encoder;
  const size_t offset = encoder.EncodeFuncListAttr(values, sizes, 3);

  auto buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data() + offset);

  EXPECT_EQ(aggr_attr.GetNumElements(), 3);

  FuncAttr str_a_attr = aggr_attr.GetAttributeOfType<FuncAttr>(0);
  string_view str_a = str_a_attr.GetFunctionName();
  EXPECT_EQ(str_a, a);

  FuncAttr str_b_attr = aggr_attr.GetAttributeOfType<FuncAttr>(1);
  string_view str_b = str_b_attr.GetFunctionName();
  EXPECT_EQ(str_b, b);

  FuncAttr str_c_attr = aggr_attr.GetAttributeOfType<FuncAttr>(2);
  string_view str_c = str_c_attr.GetFunctionName();
  EXPECT_EQ(str_c, c);
}

}  // namespace
}  // namespace tfrt
