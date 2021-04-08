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

// Unit test for TFRT Tensor.

#include "gtest/gtest.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

namespace tfrt {
namespace {

TEST(TensorTest, EmptyDenseHostTensorView) {
  auto allocator = CreateMallocAllocator();
  auto host_buffer = HostBuffer::CreateFromExternal(/*ptr=*/nullptr, /*size=*/0,
                                                    [](void*, size_t) {});
  DenseHostTensor tensor(TensorMetadata(DType(DType::I32), /*shape=*/{}),
                         std::move(host_buffer));
  DHTArrayView<int32_t> view(&tensor);

  EXPECT_TRUE(view.Elements().empty());
}

}  // namespace
}  // namespace tfrt
