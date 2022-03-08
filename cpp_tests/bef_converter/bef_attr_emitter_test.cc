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

// Unit tests for BefAttributeEmitter class.

#include "../../lib/bef_converter/mlir_to_bef/bef_attr_emitter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/init_tfrt_dialects.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace {

class BefAttrEmitterTest : public ::testing::Test {
 protected:
  BefAttrEmitterTest() {
    RegisterTFRTDialects(registry_);
    RegisterTFRTCompiledDialects(registry_);
    context_.appendDialectRegistry(registry_);
    for (const auto& dialect_name : context_.getAvailableDialects()) {
      context_.getOrLoadDialect(dialect_name);
    }
  }

  template <typename T, unsigned width,
            mlir::IntegerType::SignednessSemantics signedness>
  void TestBasicTypeEmitting(T value) {
    auto mlir_attr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, width, signedness), value);

    auto offset = emitter_.EmitAttribute(mlir_attr);
    auto buffer = emitter_.TakeResult();
    Attribute<T> attr(buffer.data() + offset);

    EXPECT_EQ(attr.get(), value);
  }

  template <typename T>
  void TestEmitting(T value, mlir::Attribute mlir_attr) {
    auto offset = emitter_.EmitAttribute(mlir_attr);
    auto buffer = emitter_.TakeResult();
    Attribute<T> attr(buffer.data() + offset);
    EXPECT_EQ(attr.get(), value);
  }

  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
  BefAttrEmitter emitter_;
};

constexpr uint8_t kTestUint8 = 1;
constexpr uint16_t kTestUint16 = 23;
constexpr uint32_t kTestUint32 = 4567;
constexpr uint64_t kTestUint64 = 12345678L;
constexpr int8_t kTestInt1 = 1;
constexpr int8_t kTestInt8 = -1;
constexpr int16_t kTestInt16 = -23;
constexpr int32_t kTestInt32 = -4567;
constexpr int64_t kTestInt64 = -12345678L;
TEST_F(BefAttrEmitterTest, EmitBasicAttributes) {
  TestBasicTypeEmitting<uint8_t, 8, mlir::IntegerType::Unsigned>(kTestUint8);
  TestBasicTypeEmitting<uint16_t, 16, mlir::IntegerType::Unsigned>(kTestUint16);
  TestBasicTypeEmitting<uint32_t, 32, mlir::IntegerType::Unsigned>(kTestUint32);
  TestBasicTypeEmitting<uint64_t, 64, mlir::IntegerType::Unsigned>(kTestUint64);
  TestBasicTypeEmitting<int8_t, 1, mlir::IntegerType::Signed>(kTestInt1);
  TestBasicTypeEmitting<int8_t, 8, mlir::IntegerType::Signed>(kTestInt8);
  TestBasicTypeEmitting<int16_t, 16, mlir::IntegerType::Signed>(kTestInt16);
  TestBasicTypeEmitting<int32_t, 32, mlir::IntegerType::Signed>(kTestInt32);
  TestBasicTypeEmitting<int64_t, 64, mlir::IntegerType::Signed>(kTestInt64);
}

constexpr float kTestFloat = 3.14;
constexpr double kTestDouble = 3.141592;
TEST_F(BefAttrEmitterTest, EmitFloatAttributes) {
  mlir::Builder builder(&context_);
  TestEmitting(kTestFloat,
               mlir::FloatAttr::get(builder.getF32Type(), kTestFloat));
  TestEmitting(kTestDouble,
               mlir::FloatAttr::get(builder.getF64Type(), kTestDouble));
}

constexpr char kTestString[] = "Hello, TFRT";
TEST_F(BefAttrEmitterTest, EmitStringAttribute) {
  auto mlir_attr = mlir::StringAttr::get(&context_, kTestString);

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  StringAttribute attr(buffer.data() + offset);
  string_view sv = attr.get();

  EXPECT_EQ(sv.size(), strlen(kTestString));
  EXPECT_EQ(sv, kTestString);
}

TEST_F(BefAttrEmitterTest, EmitI32TypeAttribute) {
  mlir::Builder builder(&context_);
  auto mlir_attr = mlir::TypeAttr::get(builder.getIntegerType(32));

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  Attribute<uint8_t> attr(buffer.data() + offset);

  EXPECT_EQ(attr.get(), static_cast<uint8_t>(DType::I32));
}

TEST_F(BefAttrEmitterTest, EmitF64TypeAttribute) {
  mlir::Builder builder(&context_);
  auto mlir_attr = mlir::TypeAttr::get(builder.getF64Type());

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  Attribute<uint8_t> attr(buffer.data() + offset);

  EXPECT_EQ(attr.get(), static_cast<uint8_t>(DType::F64));
}

constexpr int64_t kTestShape[] = {1, 2, 3};
constexpr int kTestShapeRank = sizeof(kTestShape) / sizeof(int64_t);
TEST_F(BefAttrEmitterTest, EmitShapeAttribute) {
  mlir::Builder builder(&context_);
  auto mlir_attr =
      tfrt::corert::ShapeAttr::get(builder.getContext(), kTestShape);

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  ShapeAttr attr(buffer.data() + offset);

  EXPECT_TRUE(attr.HasRank());
  EXPECT_EQ(attr.GetRank(), kTestShapeRank);

  EXPECT_THAT(attr.GetShape(),
              ::testing::ContainerEq(llvm::makeArrayRef(kTestShape, 3)));
}

TEST_F(BefAttrEmitterTest, EmitUnrankedShapeAttribute) {
  mlir::Builder builder(&context_);
  auto mlir_attr = tfrt::corert::ShapeAttr::get(builder.getContext());
  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  ShapeAttr attr(buffer.data() + offset);

  EXPECT_FALSE(attr.HasRank());
}

constexpr int32_t kTestI32Array[] = {1, 2, 3, 4};
constexpr int kTestI32ArraySize = sizeof(kTestI32Array) / sizeof(int32_t);
TEST_F(BefAttrEmitterTest, EmitI32ArrayAttribute) {
  llvm::SmallVector<mlir::Attribute, kTestI32ArraySize> elements;
  elements.reserve(kTestI32ArraySize);
  for (int idx = 0; idx < kTestI32ArraySize; ++idx) {
    elements.push_back(mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context_, 32, mlir::IntegerType::Signed),
        kTestI32Array[idx]));
  }
  auto mlir_attr = mlir::ArrayAttr::get(&context_, elements);

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  ArrayAttribute<int32_t> attr(buffer.data() + offset);

  EXPECT_THAT(attr.data(), ::testing::ContainerEq(llvm::makeArrayRef(
                               kTestI32Array, kTestI32ArraySize)));
}

constexpr double kTestF64Array[] = {1.1, 2.2, 3.3, 4.4, 5.5};
constexpr int kTestF64ArraySize = sizeof(kTestF64Array) / sizeof(double);
TEST_F(BefAttrEmitterTest, EmitF64ArrayAttribute) {
  mlir::Builder builder(&context_);
  llvm::SmallVector<mlir::Attribute, kTestF64ArraySize> elements;
  elements.reserve(kTestF64ArraySize);
  for (int idx = 0; idx < kTestF64ArraySize; ++idx) {
    elements.push_back(
        mlir::FloatAttr::get(builder.getF64Type(), kTestF64Array[idx]));
  }
  auto mlir_attr = mlir::ArrayAttr::get(&context_, elements);

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  ArrayAttribute<double> attr(buffer.data() + offset);

  EXPECT_THAT(attr.data(), ::testing::ContainerEq(llvm::makeArrayRef(
                               kTestF64Array, kTestF64ArraySize)));
}

TEST_F(BefAttrEmitterTest, EncodeDenseAttribute) {
  auto host_context = CreateHostContext();

  auto dht_create_res = DenseHostTensor::CreateUninitialized<float>(
      TensorShape({1, 2}), host_context.get());
  ASSERT_TRUE(dht_create_res.hasValue());
  DenseHostTensor dht(std::move(*dht_create_res));
  MutableDHTArrayView<float> tensor_view(&dht);
  tensor_view.Fill(1.5f);

  mlir::Builder builder(&context_);
  const auto& md = dht.metadata();

  llvm::SmallVector<int64_t, 4> shape;
  for (int i = 0; i < md.shape.GetRank(); ++i) {
    shape.push_back(md.shape.GetDimensionSize(i));
  }
  auto type = mlir::RankedTensorType::get(shape, builder.getF32Type());

  llvm::SmallVector<mlir::Attribute, 8> elements;
  for (auto element : tensor_view.Elements()) {
    elements.push_back(mlir::FloatAttr::get(builder.getF32Type(), element));
  }
  auto mlir_attr = mlir::DenseElementsAttr::get(type, elements);

  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();
  DenseAttr attr(buffer.data() + offset);

  EXPECT_EQ(attr.GetByteSize(), buffer.size() - offset);
  EXPECT_EQ(attr.GetNumElements(), 2);
  EXPECT_EQ(attr.dtype(), DType::F32);
  EXPECT_EQ(attr.shape()[0], 1);
  EXPECT_EQ(attr.shape()[1], 2);
  EXPECT_EQ(attr.GetElement<float>(0), 1.5f);
  EXPECT_EQ(attr.GetElement<float>(1), 1.5f);
}

constexpr int32_t kTestAggregateAttr1 = 123;
constexpr char kTestAggregateAttr2[] = "Aggregate Attribute";
constexpr float kTestAggregateAttr3 = 3.14;
TEST_F(BefAttrEmitterTest, EmitAggregateAttribute) {
  mlir::Builder builder(&context_);

  llvm::SmallVector<mlir::Attribute, 4> elements;

  elements.push_back(mlir::IntegerAttr::get(
      mlir::IntegerType::get(&context_, 32, mlir::IntegerType::Signed),
      kTestAggregateAttr1));

  elements.push_back(mlir::StringAttr::get(&context_, kTestAggregateAttr2));
  elements.push_back(
      mlir::FloatAttr::get(builder.getF32Type(), kTestAggregateAttr3));

  auto mlir_attr = mlir::ArrayAttr::get(&context_, elements);
  auto offset = emitter_.EmitAttribute(mlir_attr);
  auto buffer = emitter_.TakeResult();

  AggregateAttr attr(buffer.data() + offset);

  EXPECT_EQ(attr.GetNumElements(), 3);

  EXPECT_EQ(attr.GetElementType(0), static_cast<BEFAttributeType>(DType::I32));
  EXPECT_EQ(attr.GetAttributeOfType<Attribute<int32_t>>(0).get(),
            kTestAggregateAttr1);

  EXPECT_EQ(attr.GetElementType(1),
            static_cast<BEFAttributeType>(DType::String));
  EXPECT_EQ(attr.GetAttributeOfType<StringAttribute>(1).get(),
            kTestAggregateAttr2);

  EXPECT_EQ(attr.GetElementType(2), static_cast<BEFAttributeType>(DType::F32));
  EXPECT_EQ(attr.GetAttributeOfType<Attribute<float>>(2).get(),
            kTestAggregateAttr3);
}

}  // namespace
}  // namespace tfrt
