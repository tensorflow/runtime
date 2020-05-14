// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- op_attrs_test.cc -----------------------------------------*- C++ -*-===//
//
// This file has unit tests for tfrt::OpAttrs.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/op_attrs.h"

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace {

TEST(CpuDriverTest, DenseAttr) {
  auto host = CreateHostContext();

  // Create a DHT to serialize to a DenseAttr.
  auto dht_create_res = tfrt::DenseHostTensor::CreateUninitialized<float>(
      TensorShape({1, 2}), host.get());
  ASSERT_TRUE(dht_create_res.hasValue());
  DenseHostTensor dht(std::move(*dht_create_res));
  MutableDHTArrayView<float> tensor_view(&dht);
  tensor_view.Fill(1.0f);

  // `dense_attr_buffer` owns the underlying bytes.
  std::vector<uint8_t> dense_attr_buffer =
      SerializeDenseHostTensorToDenseAttr(dht);
  DenseAttr dense_attr(dense_attr_buffer.data());

  // OpAttrs::Set should copy the bytes.
  auto attrs = std::make_unique<tfrt::OpAttrs>();
  ASSERT_TRUE(attrs->Set("value", dense_attr));

  // OpAttrs::Get after OpAttrs::Set works.
  tfrt::DenseAttr dense_attr1;
  ASSERT_TRUE(attrs->Get("value", &dense_attr1));
  auto dht_des =
      DeserializeDenseHostTensorFromDenseAttr(dense_attr1, host.get());
  ASSERT_TRUE(!!dht_des);
  DenseHostTensor dht1(std::move(dht_des.get()));
  MutableDHTArrayView<float> tensor_view1(&dht1);
  ASSERT_EQ(tensor_view1[0], 1.0f);
  ASSERT_EQ(tensor_view1[1], 1.0f);

  // Erase original `dense_attr_buffer`.
  for (auto& i : dense_attr_buffer) i = 0;

  // OpAttrs::Get still works.
  tfrt::DenseAttr dense_attr2;
  ASSERT_TRUE(attrs->Get("value", &dense_attr2));
  auto dht_des2 =
      DeserializeDenseHostTensorFromDenseAttr(dense_attr2, host.get());
  ASSERT_TRUE(!!dht_des2);
  DenseHostTensor dht2(std::move(dht_des2.get()));
  MutableDHTArrayView<float> tensor_view2(&dht2);
  ASSERT_EQ(tensor_view2[0], 1.0f);
  ASSERT_EQ(tensor_view2[1], 1.0f);

  // OpAttrs::Freeze puts a copy on the heap.
  tfrt::OpAttrsRef frozen_attrs = attrs->freeze();

  // Deallocate the original attrs.
  attrs.reset();

  // OpAttrsRef::Get on the frozen attrs still works.
  tfrt::DenseAttr dense_attr3;
  ASSERT_TRUE(frozen_attrs.Get("value", &dense_attr3));
  auto dht_des3 =
      DeserializeDenseHostTensorFromDenseAttr(dense_attr3, host.get());
  ASSERT_TRUE(!!dht_des3);
  DenseHostTensor dht3(std::move(dht_des3.get()));
  MutableDHTArrayView<float> tensor_view3(&dht3);
  ASSERT_EQ(tensor_view3[0], 1.0f);
  ASSERT_EQ(tensor_view3[1], 1.0f);
}

TEST(OpAttrsTest, Array) {
  std::vector<float> values_float = {true, false};
  ArrayRef<float> values_float_ref(values_float);
  std::vector<int> values_int = {123};
  ArrayRef<int> values_int_ref(values_int);

  tfrt::OpAttrs opattrs;
  ASSERT_TRUE(opattrs.SetArray<float>("foo", values_float));
  ASSERT_TRUE(opattrs.SetArray<int>("bar", values_int));
  tfrt::OpAttrsRef opattrs_ref(opattrs);

  tfrt::ArrayRef<float> out1, out2, out3, out4;
  ASSERT_TRUE(opattrs.GetArray<float>("foo", &out1));
  ASSERT_TRUE(opattrs_ref.GetArray<float>("foo", &out4));
  ASSERT_EQ(out1, values_float_ref);
  ASSERT_EQ(out4, values_float_ref);
  // Check attribute has incorrect type (bar is int array)
  ASSERT_FALSE(opattrs.GetArray<float>("bar", &out2));
  // Check attribute doesn't exist
  ASSERT_FALSE(opattrs.GetArray<float>("baz", &out3));
}

TEST(OpAttrsTest, ArrayAsserting) {
  std::vector<int32_t> values = {34, 45};
  ArrayRef<int32_t> values_ref(values);
  std::vector<int32_t> empty;
  ArrayRef<int32_t> empty_ref(empty);

  tfrt::OpAttrs opattrs;
  ASSERT_TRUE(opattrs.SetArray<int32_t>("foo", values));
  ASSERT_TRUE(opattrs.SetArray<int32_t>("bar", empty));
  tfrt::OpAttrsRef opattrs_ref(opattrs);

  ASSERT_EQ(opattrs.GetArrayAsserting<int32_t>("foo"), values_ref);
  ASSERT_EQ(opattrs.GetArrayAsserting<int32_t>("bar"), empty_ref);
  ASSERT_EQ(opattrs_ref.GetArrayAsserting<int32_t>("foo"), values_ref);
  ASSERT_EQ(opattrs_ref.GetArrayAsserting<int32_t>("bar"), empty_ref);
}
}  // namespace
}  // namespace tfrt
