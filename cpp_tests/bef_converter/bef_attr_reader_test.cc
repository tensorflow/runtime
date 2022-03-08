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

// Unit tests for BefAttrReader class.

#include "../../lib/bef_converter/bef_to_mlir/bef_attr_reader.h"

#include <memory>

#include "../../lib/bef_converter/mlir_to_bef/bef_attr_emitter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

class BefAttrReaderTest : public ::testing::Test {
 protected:
  BefAttrReaderTest() {
    RegisterTFRTDialects(registry_);
    RegisterTFRTCompiledDialects(registry_);
    context_.appendDialectRegistry(registry_);
    for (const auto& dialect_name : context_.getAvailableDialects()) {
      context_.getOrLoadDialect(dialect_name);
    }
  }

  template <typename T, DType dtype>
  void TestIntegerAttribute(T value) {
    BefAttrEncoder encoder;
    const auto offset = encoder.EncodeAttr(value);
    auto buffer = encoder.TakeResult();
    BefAttrReader reader(buffer, &context_);

    const auto attribute_type = static_cast<BEFAttributeType>(dtype);

    auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
    EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

    EXPECT_EQ(static_cast<T>(mlir_attr.template cast<mlir::IntegerAttr>()
                                 .getValue()
                                 .getLimitedValue()),
              value);
  }

  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
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

TEST_F(BefAttrReaderTest, ReadIntegerAttributes) {
  TestIntegerAttribute<uint8_t, DType::UI8>(kTestUint8);
  TestIntegerAttribute<uint16_t, DType::UI16>(kTestUint16);
  TestIntegerAttribute<uint32_t, DType::UI32>(kTestUint32);
  TestIntegerAttribute<uint64_t, DType::UI64>(kTestUint64);
  TestIntegerAttribute<int8_t, DType::I1>(kTestInt1);
  TestIntegerAttribute<int8_t, DType::I8>(kTestInt8);
  TestIntegerAttribute<int16_t, DType::I16>(kTestInt16);
  TestIntegerAttribute<int32_t, DType::I32>(kTestInt32);
  TestIntegerAttribute<int64_t, DType::I64>(kTestInt64);
}

constexpr float kTestFloat = -3.14;
TEST_F(BefAttrReaderTest, ReadF32Attribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeAttr<float>(kTestFloat);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = static_cast<BEFAttributeType>(DType::F32);

  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  EXPECT_EQ(static_cast<float>(
                mlir_attr.cast<mlir::FloatAttr>().getValue().convertToFloat()),
            kTestFloat);
}

constexpr double kTestDeouble = -3.141592;
TEST_F(BefAttrReaderTest, ReadF64Attribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeAttr<double>(kTestDeouble);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = static_cast<BEFAttributeType>(DType::F64);

  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  EXPECT_EQ(static_cast<double>(
                mlir_attr.cast<mlir::FloatAttr>().getValue().convertToDouble()),
            kTestDeouble);
}

constexpr char kTestString[] = "Hello, World";
TEST_F(BefAttrReaderTest, ReadStringAttribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeStringAttr(kTestString);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = static_cast<BEFAttributeType>(DType::String);

  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  EXPECT_EQ(mlir_attr.cast<mlir::StringAttr>().getValue(), kTestString);
}

TEST_F(BefAttrReaderTest, ReadI32TypeAttribute) {
  mlir::Builder builder(&context_);
  auto mlir_attr = mlir::TypeAttr::get(builder.getIntegerType(32));

  BefAttrEmitter emitter;
  const auto offset = emitter.EmitAttribute(mlir_attr);
  auto buffer = emitter.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kType;

  auto read_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  EXPECT_EQ(read_attr.cast<mlir::TypeAttr>().getType(), mlir_attr.getType());
}

constexpr int64_t kTestShape[] = {1, 2, 3};
constexpr int kTestShapeRank = sizeof(kTestShape) / sizeof(int64_t);
TEST_F(BefAttrReaderTest, ReadRankedShapeAttribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeRankedShapeAttr(kTestShape);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kShape;

  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  auto shape_attr = mlir_attr.cast<tfrt::corert::ShapeAttr>();
  EXPECT_TRUE(shape_attr.hasRank());
  EXPECT_EQ(shape_attr.getRank(), kTestShapeRank);
  for (int i = 0; i < kTestShapeRank; ++i) {
    EXPECT_EQ(shape_attr.getShape()[i], kTestShape[i]);
  }
}

TEST_F(BefAttrReaderTest, ReadUnrankedShapeAttribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeUnrankedShapeAttr();
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kShape;

  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  auto shape_attr = mlir_attr.cast<tfrt::corert::ShapeAttr>();
  EXPECT_FALSE(shape_attr.hasRank());
}

constexpr int32_t kTestI32Array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
constexpr int kTestI32ArraySize = sizeof(kTestI32Array) / sizeof(int32_t);
TEST_F(BefAttrReaderTest, ReadI32ArrayAttribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeArrayAttr<int32_t>(kTestI32Array);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kI32Array;
  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  auto array_attr = mlir_attr.cast<mlir::ArrayAttr>().getValue();

  EXPECT_EQ(array_attr.size(), kTestI32ArraySize);
  for (int idx = 0; idx < kTestI32ArraySize; ++idx) {
    EXPECT_EQ(
        array_attr[idx].cast<mlir::IntegerAttr>().getValue().getLimitedValue(),
        kTestI32Array[idx]);
  }
}

constexpr double kTestF64Array[] = {1.1, 2.2, 3.3, 4.4, 5.5};
constexpr int kTestF64ArraySize = sizeof(kTestF64Array) / sizeof(double);
TEST_F(BefAttrReaderTest, ReadF64ArrayAttribute) {
  BefAttrEncoder encoder;
  const auto offset = encoder.EncodeArrayAttr<double>(kTestF64Array);
  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kF64Array;
  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  auto array_attr = mlir_attr.cast<mlir::ArrayAttr>().getValue();

  EXPECT_EQ(array_attr.size(), kTestF64ArraySize);
  for (int idx = 0; idx < kTestF64ArraySize; ++idx) {
    EXPECT_EQ(
        array_attr[idx].cast<mlir::FloatAttr>().getValue().convertToDouble(),
        kTestF64Array[idx]);
  }
}

TEST_F(BefAttrReaderTest, ReadDenseAttribute) {
  auto host_context = CreateHostContext();

  auto dht_create_res = DenseHostTensor::CreateUninitialized<float>(
      TensorShape({1, 2}), host_context.get());
  ASSERT_TRUE(dht_create_res.hasValue());
  DenseHostTensor dht(std::move(*dht_create_res));
  MutableDHTArrayView<float> tensor_view(&dht);
  tensor_view.Fill(1.5f);

  const auto& md = dht.metadata();
  llvm::SmallVector<int64_t, 2> shape;
  for (int i = 0; i < md.shape.GetRank(); ++i) {
    shape.push_back(md.shape.GetDimensionSize(i));
  }

  BefAttrEncoder encoder;
  auto offset = encoder.EncodeDenseAttr(
      md.dtype, shape,
      llvm::makeArrayRef(static_cast<const uint8_t*>(dht.data()),
                         dht.DataSizeInBytes()));

  auto buffer = encoder.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kDense;
  auto mlir_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(mlir_attr), attribute_type);

  auto dense_attr = mlir_attr.cast<mlir::DenseElementsAttr>();
  const auto shaped_type = dense_attr.getType();

  EXPECT_EQ(
      BefAttrEmitter::ConvertMlirTypeToDType(shaped_type.getElementType()),
      DType::F32);
  EXPECT_EQ(shaped_type.getShape().size(), 2);
  EXPECT_EQ(shaped_type.getShape()[0], 1);
  EXPECT_EQ(shaped_type.getShape()[1], 2);

  for (auto element : dense_attr.getValues<mlir::Attribute>()) {
    EXPECT_EQ(element.cast<mlir::FloatAttr>().getValue().convertToFloat(),
              1.5f);
  }
}

constexpr int32_t kTestAggregateAttr1 = 123;
constexpr char kTestAggregateAttr2[] = "Aggregate Attribute";
constexpr float kTestAggregateAttr3 = 3.14;
TEST_F(BefAttrReaderTest, EmitAggregateAttribute) {
  mlir::Builder builder(&context_);

  llvm::SmallVector<mlir::Attribute, 4> elements;

  elements.push_back(mlir::IntegerAttr::get(
      mlir::IntegerType::get(&context_, 32, mlir::IntegerType::Signed),
      kTestAggregateAttr1));

  elements.push_back(mlir::StringAttr::get(&context_, kTestAggregateAttr2));
  elements.push_back(
      mlir::FloatAttr::get(builder.getF32Type(), kTestAggregateAttr3));

  auto mlir_attr = mlir::ArrayAttr::get(&context_, elements);

  BefAttrEmitter emitter;
  const auto offset = emitter.EmitAttribute(mlir_attr);

  auto buffer = emitter.TakeResult();
  BefAttrReader reader(buffer, &context_);

  const auto attribute_type = BEFAttributeType::kAggregate;
  auto read_attr = reader.ReadAttribute(attribute_type, offset);
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(read_attr), attribute_type);

  auto aggregate_attr = read_attr.cast<mlir::ArrayAttr>();
  EXPECT_EQ(aggregate_attr.size(), 3);

  auto first = aggregate_attr[0];
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(first),
            static_cast<BEFAttributeType>(DType::I32));
  EXPECT_EQ(static_cast<int32_t>(
                first.cast<mlir::IntegerAttr>().getValue().getLimitedValue()),
            kTestAggregateAttr1);

  auto second = aggregate_attr[1];
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(second),
            static_cast<BEFAttributeType>(DType::String));
  EXPECT_EQ(second.cast<mlir::StringAttr>().getValue(), kTestAggregateAttr2);

  auto third = aggregate_attr[2];
  EXPECT_EQ(BefAttrEmitter::GetBefAttributeType(third),
            static_cast<BEFAttributeType>(DType::F32));
  EXPECT_EQ(static_cast<float>(
                third.cast<mlir::FloatAttr>().getValue().convertToFloat()),
            kTestAggregateAttr3);
}

}  // namespace
}  // namespace tfrt
