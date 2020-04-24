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

//===- dense_host_tensor_test.cc --------------------------------*- C++ -*-===//
//
// Unit test for DenseHostTensor.
//
//===----------------------------------------------------------------------===//

#include "tfrt/tensor/dense_host_tensor.h"

#include <complex>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"

namespace tfrt {
namespace {

TEST(DenseHostTensorTest, FillWithComplex64Type) {
  auto host = CreateHostContext();

  auto dht_create_res_a =
      tfrt::DenseHostTensor::CreateUninitialized<std::complex<float>>(
          TensorShape({1, 2}), host.get());
  ASSERT_TRUE(dht_create_res_a.hasValue());
  DenseHostTensor dht_a(std::move(*dht_create_res_a));
  MutableDHTArrayView<std::complex<float>> tensor_view_a(&dht_a);
  tensor_view_a.Fill({1.0, -2.0});

  auto dht_create_res_b =
      tfrt::DenseHostTensor::CreateUninitialized<std::complex<float>>(
          TensorShape({1, 2}), host.get());
  ASSERT_TRUE(dht_create_res_b.hasValue());
  DenseHostTensor dht_b(std::move(*dht_create_res_b));
  MutableDHTArrayView<std::complex<float>> tensor_view_b(&dht_b);
  tensor_view_b.Fill({1.0, -2.0});
  EXPECT_TRUE(AllElementsClose(DHTArrayView<std::complex<float>>(&dht_a),
                               DHTArrayView<std::complex<float>>(&dht_b)));
}

}  // namespace
}  // namespace tfrt
