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

#include "tfrt/tensor/btf.h"

#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/tensor/btf_util.h"

namespace tfrt {
namespace btf {
namespace {

TEST(BTFTest, ToDTypeKind) {
  EXPECT_EQ(ToDTypeKind(TensorDType::kInt8), DType::I8);
  EXPECT_EQ(ToDTypeKind(TensorDType::kInt16), DType::I16);
  EXPECT_EQ(ToDTypeKind(TensorDType::kInt32), DType::I32);
  EXPECT_EQ(ToDTypeKind(TensorDType::kInt64), DType::I64);
  EXPECT_EQ(ToDTypeKind(TensorDType::kFloat32), DType::F32);
  EXPECT_EQ(ToDTypeKind(TensorDType::kFloat64), DType::F64);
  EXPECT_EQ(ToDTypeKind(TensorDType::kUInt8), DType::UI8);
  EXPECT_EQ(ToDTypeKind(TensorDType::kUInt16), DType::UI16);
  EXPECT_EQ(ToDTypeKind(TensorDType::kUInt32), DType::UI32);
  EXPECT_EQ(ToDTypeKind(TensorDType::kUInt64), DType::UI64);
}

TEST(BTFTest, ToTensorDType) {
  EXPECT_EQ(ToTensorDType(DType::I8).get(), TensorDType::kInt8);
  EXPECT_EQ(ToTensorDType(DType::I16).get(), TensorDType::kInt16);
  EXPECT_EQ(ToTensorDType(DType::I32).get(), TensorDType::kInt32);
  EXPECT_EQ(ToTensorDType(DType::I64).get(), TensorDType::kInt64);
  EXPECT_EQ(ToTensorDType(DType::F32).get(), TensorDType::kFloat32);
  EXPECT_EQ(ToTensorDType(DType::F64).get(), TensorDType::kFloat64);
  EXPECT_EQ(ToTensorDType(DType::UI8).get(), TensorDType::kUInt8);
  EXPECT_EQ(ToTensorDType(DType::UI16).get(), TensorDType::kUInt16);
  EXPECT_EQ(ToTensorDType(DType::UI32).get(), TensorDType::kUInt32);
  EXPECT_EQ(ToTensorDType(DType::UI64).get(), TensorDType::kUInt64);
}

TEST(BTFTest, ToTensorDTypeUnsupported) {
  auto error = ToTensorDType(DType::Complex128).takeError();
  EXPECT_FALSE(error.success());
}

template <typename EnumT>
void CheckEnumStr(EnumT e, const char* str) {
  EXPECT_EQ(StrCat(e), str);
}

TEST(BTFTest, TensorDTypeToRawStream) {
  CheckEnumStr(TensorDType::kInt8, "i8");
  CheckEnumStr(TensorDType::kInt16, "i16");
  CheckEnumStr(TensorDType::kInt32, "i32");
  CheckEnumStr(TensorDType::kInt64, "i64");
  CheckEnumStr(TensorDType::kFloat32, "f32");
  CheckEnumStr(TensorDType::kFloat64, "f64");
  CheckEnumStr(TensorDType::kUInt8, "ui8");
  CheckEnumStr(TensorDType::kUInt16, "ui16");
  CheckEnumStr(TensorDType::kUInt32, "ui32");
  CheckEnumStr(TensorDType::kUInt64, "ui64");
}

TEST(BTFTest, TensorLayoutToRawStream) {
  CheckEnumStr(TensorLayout::kRMD, "Row-Major Dense tensor");
  CheckEnumStr(TensorLayout::kCOO_EXPERIMENTAL,
               "COOrdinate list sparse tensor");
}

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
}  // namespace btf
}  // namespace tfrt
