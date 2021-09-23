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

// Unit test for DenseHostTensor.

#include "tfrt/tensor/dense_host_tensor.h"

#include <complex>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/test_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"

namespace tfrt {
namespace {

TEST(DenseHostTensorTest, DefaultConstructible) {
  tfrt::DenseHostTensor dht;
  tfrt::DenseHostTensor other = dht.CopyRef();
  EXPECT_EQ(dht.dtype(), DType::Invalid);
  EXPECT_EQ(other.dtype(), DType::Invalid);
}

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
  EXPECT_TRUE(TensorApproxEqual<std::complex<float>>(dht_a, dht_b));
}

TEST(DenseHostTensorTest, FillWithComplex128Type) {
  auto host = CreateHostContext();

  auto dht_create_res_a =
      tfrt::DenseHostTensor::CreateUninitialized<std::complex<double>>(
          TensorShape({1, 2}), host.get());
  ASSERT_TRUE(dht_create_res_a.hasValue());
  DenseHostTensor dht_a(std::move(*dht_create_res_a));
  MutableDHTArrayView<std::complex<double>> tensor_view_a(&dht_a);
  tensor_view_a.Fill({1.0, -2.0});

  auto dht_create_res_b =
      tfrt::DenseHostTensor::CreateUninitialized<std::complex<double>>(
          TensorShape({1, 2}), host.get());
  ASSERT_TRUE(dht_create_res_b.hasValue());
  DenseHostTensor dht_b(std::move(*dht_create_res_b));
  MutableDHTArrayView<std::complex<double>> tensor_view_b(&dht_b);
  tensor_view_b.Fill({1.0, -2.0});
  EXPECT_TRUE(TensorApproxEqual<std::complex<double>>(dht_a, dht_b));
}

TEST(DenseHostTensorTest, FillWithBF16Type) {
  // Test DType::BF16. The bf16 is a placeholder but not the actual
  // implementation of brain float 16.
  auto host = CreateHostContext();

  auto dht_create_res_a = tfrt::DenseHostTensor::CreateUninitialized<bf16>(
      TensorShape({1, 1}), host.get());
  ASSERT_TRUE(dht_create_res_a.hasValue());
  DenseHostTensor dht_a(std::move(*dht_create_res_a));
  MutableDHTArrayView<bf16> tensor_view_a(&dht_a);
  tensor_view_a.Fill(bf16{static_cast<uint16_t>(1.0)});

  auto dht_create_res_b = tfrt::DenseHostTensor::CreateUninitialized<bf16>(
      TensorShape({1, 1}), host.get());
  ASSERT_TRUE(dht_create_res_b.hasValue());
  DenseHostTensor dht_b(std::move(*dht_create_res_b));
  MutableDHTArrayView<bf16> tensor_view_b(&dht_b);
  tensor_view_b.Fill(bf16{static_cast<uint16_t>(1.0)});

  // Compare the buffer value, which is uint_16.
  EXPECT_TRUE(tensor_view_a[0].value == tensor_view_b[0].value);
}

TEST(DenseHostTensorSharedTest, FillWithComplex64Type) {
  // Creates a HostBuffer that is shared between 2 distinct DenseHostTensors.
  // Validates the DenseHostTensor values against the parent HostBuffer.
  auto host = CreateHostContext();
  auto parent_buffer = tfrt::HostBuffer::CreateUninitialized(
      /*size=*/16,
      /*alignment=*/sizeof(std::complex<float>), host->allocator());

  // Create dht_a from 0 bytes to 7 bytes in the buffer.
  auto host_buffer_a = tfrt::HostBuffer::CreateFromExternal(parent_buffer,
                                                            /*offset=*/0,
                                                            /*size=*/8);
  auto dht_a = tfrt::DenseHostTensor(
      TensorMetadata(GetDType<std::complex<float>>(), TensorShape({1, 1})),
      std::move(host_buffer_a));
  MutableDHTArrayView<std::complex<float>> tensor_view_a(&dht_a);
  tensor_view_a.Fill({1.0, -2.0});

  // Create dht_b from 8 bytes to 15 bytes in the buffer.
  auto host_buffer_b = tfrt::HostBuffer::CreateFromExternal(parent_buffer,
                                                            /*offset=*/8,
                                                            /*size=*/8);
  auto dht_b = tfrt::DenseHostTensor(
      TensorMetadata(GetDType<std::complex<float>>(), TensorShape({1, 1})),
      std::move(host_buffer_b));
  MutableDHTArrayView<std::complex<float>> tensor_view_b(&dht_b);
  tensor_view_b.Fill({3.0, -4.0});

  // Compare the values of the parent buffer with the slices.
  ASSERT_EQ(parent_buffer->size(), 16);
  ASSERT_EQ(dht_a.DataSizeInBytes(), 8);
  ASSERT_EQ(dht_b.DataSizeInBytes(), 8);

  auto parent_data = static_cast<std::complex<float> *>(parent_buffer->data());
  auto dht_a_data = static_cast<std::complex<float> *>(dht_a.data());
  auto dht_b_data = static_cast<std::complex<float> *>(dht_b.data());
  ASSERT_EQ(parent_data[0], dht_a_data[0]);
  ASSERT_EQ(parent_data[1], dht_b_data[0]);
}

TEST(DenseHostTensorSharedTest, FillWithInt32Type) {
  // Creates a HostBuffer that is shared between 3 overlapping DenseHostTensors.
  // A is 0-7 bytes, B is 8-15 bytes, C is 8-11 bytes.
  auto host = CreateHostContext();
  auto parent_buffer = tfrt::HostBuffer::CreateUninitialized(
      /*size=*/16,
      /*alignment=*/sizeof(int), host->allocator());

  // Create dht_a from 0 bytes to 7 bytes in the buffer.
  auto host_buffer_a = tfrt::HostBuffer::CreateFromExternal(parent_buffer,
                                                            /*offset=*/0,
                                                            /*size=*/8);
  auto dht_a = tfrt::DenseHostTensor(
      TensorMetadata(GetDType<int>(), TensorShape({1, 2})),
      std::move(host_buffer_a));
  MutableDHTArrayView<int> tensor_view_a(&dht_a);
  tensor_view_a.Fill(1.0);

  auto dht_a_data = static_cast<int *>(dht_a.data());
  ASSERT_EQ(dht_a_data[0], 1.0);
  ASSERT_EQ(dht_a_data[1], 1.0);

  // Create dht_b from 8 bytes to 15 bytes in the buffer.
  auto host_buffer_b = tfrt::HostBuffer::CreateFromExternal(parent_buffer,
                                                            /*offset=*/8,
                                                            /*size=*/8);
  auto dht_b = tfrt::DenseHostTensor(
      TensorMetadata(GetDType<int>(), TensorShape({1, 2})),
      std::move(host_buffer_b));
  MutableDHTArrayView<int> tensor_view_b(&dht_b);
  tensor_view_b.Fill(3.0);

  auto dht_b_data = static_cast<int *>(dht_b.data());
  ASSERT_EQ(dht_b_data[0], 3.0);
  ASSERT_EQ(dht_b_data[1], 3.0);

  // Create dht_c from 8 bytes to 11 bytes.
  auto host_buffer_c = tfrt::HostBuffer::CreateFromExternal(parent_buffer,
                                                            /*offset=*/8,
                                                            /*size=*/4);
  auto dht_c = tfrt::DenseHostTensor(
      TensorMetadata(GetDType<int>(), TensorShape({1, 1})),
      std::move(host_buffer_c));
  MutableDHTArrayView<int> tensor_view_c(&dht_c);
  tensor_view_c.Fill(-1.0);

  // Check the values of all of the buffers.
  ASSERT_EQ(parent_buffer->size(), 16);
  ASSERT_EQ(dht_a.DataSizeInBytes(), 8);
  ASSERT_EQ(dht_b.DataSizeInBytes(), 8);
  ASSERT_EQ(dht_c.DataSizeInBytes(), 4);

  dht_a_data = static_cast<int *>(dht_a.data());
  dht_b_data = static_cast<int *>(dht_b.data());
  auto dht_c_data = static_cast<int *>(dht_c.data());
  ASSERT_EQ(dht_a_data[0], 1.0);
  ASSERT_EQ(dht_a_data[1], 1.0);
  ASSERT_EQ(dht_c_data[0], -1.0);
  ASSERT_EQ(dht_b_data[0], -1.0);
  ASSERT_EQ(dht_b_data[1], 3.0);
}

}  // namespace
}  // namespace tfrt
