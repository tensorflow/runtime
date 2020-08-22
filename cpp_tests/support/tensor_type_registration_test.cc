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

//===- tensor_type_registration.cc ------------------------------*- C++ -*-===//
//
// Unit test for TensorTypeRegistration.
//
//===----------------------------------------------------------------------===//
#include "tfrt/tensor/tensor_type_registration.h"

#include <complex>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"

namespace tfrt {

TensorType test_matters_tensor = RegisterStaticTensorType("TestMattersTensor");

TEST(TensorTypeRegistrationTest, Macro) {
  TensorType tensor_type = GetStaticTensorType("TestMattersTensor");
  EXPECT_TRUE(tensor_type == test_matters_tensor);
  EXPECT_EQ(tensor_type.name(), "TestMattersTensor");
}

}  // namespace tfrt
