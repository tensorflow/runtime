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

// Unit test for TFRT BEFAttrEncoder.

#include "tfrt/bef_converter/bef_attr_encoder.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {

TEST(BEFAttrEncoderTest, EncodeZeroShape) {
  int64_t dims[1];

  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeRankedShapeAttr(llvm::makeArrayRef(dims, 0)));

  AlignedBuffer<8> buf = encoder.TakeResult();
  RankedShapeAttr shape_attr(buf.data());

  ASSERT_EQ(shape_attr.size(), sizeof(BEFShapeAttr));
  ASSERT_EQ(shape_attr.GetRank(), 0);
  ArrayRef<int64_t> shape = shape_attr.GetShape();
  ASSERT_EQ(shape.size(), 0);
}

TEST(BEFAttrEncoderTest, EncodeUnrankedShape) {
  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeUnrankedShapeAttr());

  AlignedBuffer<8> buf = encoder.TakeResult();
  ShapeAttr shape_attr(buf.data());

  ASSERT_FALSE(shape_attr.HasRank());
}

TEST(BEFAttrEncoderTest, EncodeRankedShape) {
  int64_t dims[3] = {1, 2, 3};

  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeRankedShapeAttr(llvm::makeArrayRef(dims, 3)));

  AlignedBuffer<8> buf = encoder.TakeResult();
  RankedShapeAttr shape_attr(buf.data());

  ASSERT_EQ(shape_attr.GetRank(), 3);

  ArrayRef<int64_t> shape = shape_attr.GetShape();
  ASSERT_EQ(shape.size(), 3);
  ASSERT_EQ(shape[0], 1);
  ASSERT_EQ(shape[1], 2);
  ASSERT_EQ(shape[2], 3);
}

TEST(BEFAttrEncoderTest, EncodeShapeList) {
  const int64_t a[1] = {1};
  const int64_t b[2] = {2, 3};
  const int64_t c[3] = {4, 5, 6};
  const int64_t* dims[4] = {a, b, c, nullptr};

  int sizes[4] = {1, 2, 3, -1};

  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeShapeListAttr(dims, sizes, 4));

  AlignedBuffer<8> buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data());

  ASSERT_EQ(aggr_attr.GetNumElements(), 4);

  RankedShapeAttr shape_a = aggr_attr.GetAttributeOfType<RankedShapeAttr>(0);
  ArrayRef<int64_t> elems_array_a = shape_a.GetShape();
  ASSERT_EQ(elems_array_a.size(), 1);
  ASSERT_EQ(elems_array_a[0], 1);

  RankedShapeAttr shape_b = aggr_attr.GetAttributeOfType<RankedShapeAttr>(1);
  ArrayRef<int64_t> elems_array_b = shape_b.GetShape();
  ASSERT_EQ(elems_array_b.size(), 2);
  ASSERT_EQ(elems_array_b[0], 2);
  ASSERT_EQ(elems_array_b[1], 3);

  RankedShapeAttr shape_c = aggr_attr.GetAttributeOfType<RankedShapeAttr>(2);
  ArrayRef<int64_t> elems_array_c = shape_c.GetShape();
  ASSERT_EQ(elems_array_c.size(), 3);
  ASSERT_EQ(elems_array_c[0], 4);
  ASSERT_EQ(elems_array_c[1], 5);
  ASSERT_EQ(elems_array_c[2], 6);

  ShapeAttr shape_d = aggr_attr.GetAttributeOfType<ShapeAttr>(3);
  ASSERT_FALSE(shape_d.HasRank());
}

TEST(BEFAttrEncoderTest, EncodeEmptyString) {
  BefAttrEncoder encoder;
  std::string empty_string = "";
  ASSERT_TRUE(!encoder.EncodeStringAttr(string_view(empty_string.data(), 0)));

  AlignedBuffer<8> buf = encoder.TakeResult();
  StringAttr string_attr(buf.data());

  ASSERT_EQ(string_attr.GetValue().size(), 0);
}

TEST(BEFAttrEncoderTest, EncodeString) {
  BefAttrEncoder encoder;
  std::string sample_string = "tfrt";
  ASSERT_TRUE(!encoder.EncodeStringAttr(
      string_view(sample_string.data(), sample_string.size())));

  AlignedBuffer<8> buf = encoder.TakeResult();
  StringAttr string_attr(buf.data());

  string_view sv = string_attr.GetValue();

  ASSERT_EQ(sv.size(), 4);
  ASSERT_EQ(sv, "tfrt");
}

TEST(BEFAttrEncoderTest, EncodeStringList) {
  const std::string a = "hi";
  const std::string b = "tfrt";
  const std::string c = "world";

  const void* values[3] = {a.data(), b.data(), c.data()};

  size_t sizes[3] = {a.size(), b.size(), c.size()};

  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeStringListAttr(values, sizes, 3));

  AlignedBuffer<8> buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data());

  ASSERT_EQ(aggr_attr.GetNumElements(), 3);

  StringAttr str_a_attr = aggr_attr.GetAttributeOfType<StringAttr>(0);
  string_view str_a = str_a_attr.GetValue();
  ASSERT_EQ(str_a.size(), a.size());
  ASSERT_EQ(str_a, a);

  StringAttr str_b_attr = aggr_attr.GetAttributeOfType<StringAttr>(1);
  string_view str_b = str_b_attr.GetValue();
  ASSERT_EQ(str_b.size(), b.size());
  ASSERT_EQ(str_b, b);

  StringAttr str_c_attr = aggr_attr.GetAttributeOfType<StringAttr>(2);
  string_view str_c = str_c_attr.GetValue();
  ASSERT_EQ(str_c.size(), c.size());
  ASSERT_EQ(str_c, c);
}

TEST(BEFAttrEncoderTest, EncodeFuncList) {
  const std::string a = "tf";
  const std::string b = "new";
  const std::string c = "runtime";

  const void* values[3] = {a.data(), b.data(), c.data()};

  size_t sizes[3] = {a.size(), b.size(), c.size()};

  BefAttrEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeFuncListAttr(values, sizes, 3));

  AlignedBuffer<8> buf = encoder.TakeResult();
  AggregateAttr aggr_attr(buf.data());

  ASSERT_EQ(aggr_attr.GetNumElements(), 3);

  FuncAttr str_a_attr = aggr_attr.GetAttributeOfType<FuncAttr>(0);
  string_view str_a = str_a_attr.GetFunctionName();
  ASSERT_EQ(str_a.size(), a.size());
  ASSERT_EQ(str_a, a);

  FuncAttr str_b_attr = aggr_attr.GetAttributeOfType<FuncAttr>(1);
  string_view str_b = str_b_attr.GetFunctionName();
  ASSERT_EQ(str_b.size(), b.size());
  ASSERT_EQ(str_b, b);

  FuncAttr str_c_attr = aggr_attr.GetAttributeOfType<FuncAttr>(2);
  string_view str_c = str_c_attr.GetFunctionName();
  ASSERT_EQ(str_c.size(), c.size());
  ASSERT_EQ(str_c, c);
}

}  // namespace
}  // namespace tfrt
