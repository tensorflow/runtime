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

//===- bef_attr_encoder_test.cc ---------------------------------*- C++ -*-===//
//
// Unit test for TFRT BEFAttrEncoder.
//
//===----------------------------------------------------------------------===//

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

  BEFTypedAttributeEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeShapeAttr(llvm::makeArrayRef(dims, 0)));

  auto buf = encoder.TakeResult();
  tfrt::ShapeAttr shape_attr(buf.data());

  ASSERT_EQ(shape_attr.GetRank(), 0);
  auto shape = shape_attr.GetShape();
  ASSERT_EQ(shape.size(), 0);
}

TEST(BEFAttrEncoderTest, EncodeShape) {
  int64_t dims[3] = {1, 2, 3};

  BEFTypedAttributeEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeShapeAttr(llvm::makeArrayRef(dims, 3)));

  auto buf = encoder.TakeResult();
  tfrt::ShapeAttr shape_attr(buf.data());

  ASSERT_EQ(shape_attr.GetRank(), 3);

  auto shape = shape_attr.GetShape();
  ASSERT_EQ(shape.size(), 3);
  ASSERT_EQ(shape[0], 1);
  ASSERT_EQ(shape[1], 2);
  ASSERT_EQ(shape[2], 3);
}

TEST(BEFAttrEncoderTest, EncodeShapeList) {
  const int64_t a[1] = {1};
  const int64_t b[2] = {2, 3};
  const int64_t c[3] = {4, 5, 6};
  const int64_t* dims[3] = {a, b, c};

  int sizes[3] = {1, 2, 3};

  BEFTypedAttributeEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeShapeListAttr(dims, sizes, 3));

  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data());

  ASSERT_EQ(aggr_attr.GetNumElements(), 3);

  auto shape_a = aggr_attr.GetAttributeOfType<tfrt::ShapeAttr>(0);
  auto elems_array_a = shape_a.GetShape();
  ASSERT_EQ(elems_array_a.size(), 1);
  ASSERT_EQ(elems_array_a[0], 1);

  auto shape_b = aggr_attr.GetAttributeOfType<tfrt::ShapeAttr>(1);
  auto elems_array_b = shape_b.GetShape();
  ASSERT_EQ(elems_array_b.size(), 2);
  ASSERT_EQ(elems_array_b[0], 2);
  ASSERT_EQ(elems_array_b[1], 3);

  auto shape_c = aggr_attr.GetAttributeOfType<tfrt::ShapeAttr>(2);
  auto elems_array_c = shape_c.GetShape();
  ASSERT_EQ(elems_array_c.size(), 3);
  ASSERT_EQ(elems_array_c[0], 4);
  ASSERT_EQ(elems_array_c[1], 5);
  ASSERT_EQ(elems_array_c[2], 6);
}

TEST(BEFAttrEncoderTest, EncodeEmptyString) {
  BEFTypedAttributeEncoder encoder;
  std::string empty_string = "";
  ASSERT_TRUE(!encoder.EncodeStringAttr(string_view(empty_string.data(), 0)));

  auto buf = encoder.TakeResult();
  tfrt::StringAttr string_attr(buf.data());

  ASSERT_EQ(string_attr.GetValue().size(), 0);
}

TEST(BEFAttrEncoderTest, EncodeString) {
  BEFTypedAttributeEncoder encoder;
  std::string sample_string = "tfrt";
  ASSERT_TRUE(!encoder.EncodeStringAttr(
      string_view(sample_string.data(), sample_string.size())));

  auto buf = encoder.TakeResult();
  tfrt::StringAttr string_attr(buf.data());

  auto sv = string_attr.GetValue();

  ASSERT_EQ(sv.size(), 4);
  ASSERT_EQ(sv, "tfrt");
}

TEST(BEFAttrEncoderTest, EncodeStringList) {
  const std::string a = "hi";
  const std::string b = "tfrt";
  const std::string c = "world";

  const void* values[3] = {a.data(), b.data(), c.data()};

  size_t sizes[3] = {a.size(), b.size(), c.size()};

  BEFTypedAttributeEncoder encoder;
  ASSERT_TRUE(!encoder.EncodeStringListAttr(values, sizes, 3));

  auto buf = encoder.TakeResult();
  tfrt::AggregateAttr aggr_attr(buf.data());

  ASSERT_EQ(aggr_attr.GetNumElements(), 3);

  auto str_a_attr = aggr_attr.GetAttributeOfType<tfrt::StringAttr>(0);
  auto str_a = str_a_attr.GetValue();
  ASSERT_EQ(str_a.size(), a.size());
  ASSERT_EQ(str_a, a);

  auto str_b_attr = aggr_attr.GetAttributeOfType<tfrt::StringAttr>(1);
  auto str_b = str_b_attr.GetValue();
  ASSERT_EQ(str_b.size(), b.size());
  ASSERT_EQ(str_b, b);

  auto str_c_attr = aggr_attr.GetAttributeOfType<tfrt::StringAttr>(2);
  auto str_c = str_c_attr.GetValue();
  ASSERT_EQ(str_c.size(), c.size());
  ASSERT_EQ(str_c, c);
}

}  // namespace
}  // namespace tfrt
