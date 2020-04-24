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

//===- driver_test.cc -------------------------------------------*- C++ -*-===//
//
// This file is a simple example showing how to use core_runtime API to execute
// some simple ops.
//
//===----------------------------------------------------------------------===//

#include "driver.h"

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/ostream.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
namespace {

class CpuDriverTest : public testing::Test {
 protected:
  example::CoreRuntimeDriver driver_{"cpu"};
};

TEST_F(CpuDriverTest, MatmulTest) {
  tfrt::OpAttrs attrs;
  attrs.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 2});
  attrs.SetArray("values", tfrt::ArrayRef<float>{2.0});
  tfrt::TensorHandle a1;
  driver_.Execute("tfrt_test.create_dense_tensor",
                  driver_.CreateLocation(__FILE__, __LINE__), {},
                  attrs.freeze(), a1);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
  tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
  tfrt::TensorHandle a2;
  driver_.Execute("tfrt_test.matmul",
                  driver_.CreateLocation(__FILE__, __LINE__), matmul_args,
                  matmul_attrs_ref, a2);

  tfrt::OpAttrs empty_attrs;
  tfrt::OpAttrsRef empty_attrs_ref = empty_attrs.freeze();
  // This op will print the shape and the value of the result.
  tfrt::TensorHandle a2_ref = a2.CopyRef();
  driver_.Execute("tfrt_test.print", driver_.CreateLocation(__FILE__, __LINE__),
                  a2_ref, empty_attrs_ref, {});

  // Check the output tensor.
  auto a2_metadata = a2.GetAvailableMetadata();
  ASSERT_EQ(a2_metadata.shape.GetRank(), 2);
  ASSERT_EQ(a2_metadata.shape.GetDimensionSize(0), 2);
  ASSERT_EQ(a2_metadata.shape.GetDimensionSize(1), 2);

  auto a2_view =
      DHTArrayView<float>(&a2.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a2_view.Elements()[0], 8.0);
  ASSERT_EQ(a2_view.Elements()[1], 8.0);
  ASSERT_EQ(a2_view.Elements()[2], 8.0);
  ASSERT_EQ(a2_view.Elements()[3], 8.0);
}

TEST_F(CpuDriverTest, ReluTest_InputForward) {
  tfrt::OpAttrs attrs;
  attrs.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 2});
  attrs.SetArray("values", tfrt::ArrayRef<float>{2.0});
  tfrt::TensorHandle a1;
  driver_.Execute("tfrt_test.create_dense_tensor",
                  driver_.CreateLocation(__FILE__, __LINE__), {},
                  attrs.freeze(), a1);
  auto buffer_pointer =
      a1.GetAsyncTensor()->get<DenseHostTensor>().buffer().get();

  tfrt::OpAttrs empty_attrs;
  tfrt::TensorHandle a2;
  driver_.Execute("tfrt_test.relu", driver_.CreateLocation(__FILE__, __LINE__),
                  a1, empty_attrs.freeze(), a2);

  ASSERT_EQ(a2.GetAsyncTensor()->get<DenseHostTensor>().buffer().get(),
            buffer_pointer);
}

TEST_F(CpuDriverTest, MatmulWithError) {
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<ssize_t>{1, 1});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0});
  driver_.Execute("tfrt_test.create_dense_tensor",
                  driver_.CreateLocation(__FILE__, __LINE__), {},
                  attrs1.freeze(), a1);

  tfrt::OpAttrs attrs2;
  tfrt::TensorHandle a2;
  attrs2.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 1});
  attrs2.SetArray("values", tfrt::ArrayRef<float>{2.0});
  driver_.Execute("tfrt_test.create_dense_tensor",
                  driver_.CreateLocation(__FILE__, __LINE__), {},
                  attrs2.freeze(), a2);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
  // Since the two arguments do not have compatible shapes, this op will fail.
  tfrt::TensorHandle matmul_args1[2] = {a1.CopyRef(), a2.CopyRef()};
  tfrt::TensorHandle a3;
  // Point to the CreateLocation() call below.
  const int failed_line_num = __LINE__ + 2;
  driver_.Execute("tfrt_test.matmul",
                  driver_.CreateLocation(__FILE__, __LINE__), matmul_args1,
                  matmul_attrs_ref, a3);

  // This op will finish successfully.
  tfrt::TensorHandle matmul_args2[2] = {a1.CopyRef(), a1.CopyRef()};
  tfrt::TensorHandle a4;
  driver_.Execute("tfrt_test.matmul",
                  driver_.CreateLocation(__FILE__, __LINE__), matmul_args2,
                  matmul_attrs_ref, a4);

  tfrt::OpAttrs empty_attrs;
  tfrt::OpAttrsRef empty_attrs_ref = empty_attrs.freeze();
  tfrt::TensorHandle a4_ref = a4.CopyRef();
  driver_.Execute("tfrt_test.print", driver_.CreateLocation(__FILE__, __LINE__),
                  a4_ref, empty_attrs_ref, {});

  auto a4_view =
      DHTArrayView<float>(&a4.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a4_view.NumElements(), 1);
  ASSERT_EQ(a4_view.Elements()[0], 4.0);

  // Print the error of a previous op.
  ASSERT_TRUE(a3.GetAsyncTensor()->IsError());
  auto error = a3.GetAsyncTensor()->GetError();
  ASSERT_EQ(error.location->filename, "cpp_tests/core_runtime/driver_test.cc");
  ASSERT_EQ(error.location->line, failed_line_num);

  tfrt::outs() << error << "\n";
  tfrt::outs().flush();
}

TEST_F(CpuDriverTest, NoLocation) {
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<ssize_t>{1, 1});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0});
  driver_.Execute("tfrt_test.create_dense_tensor", Location(), {},
                  attrs1.freeze(), a1);

  tfrt::OpAttrs attrs2;
  tfrt::TensorHandle a2;
  attrs2.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 1});
  attrs2.SetArray("values", tfrt::ArrayRef<float>{2.0});
  driver_.Execute("tfrt_test.create_dense_tensor", Location(), {},
                  attrs2.freeze(), a2);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
  // Since the two arguments do not have compatible shapes, this op will fail.
  tfrt::TensorHandle matmul_args1[2] = {a1.CopyRef(), a2.CopyRef()};
  tfrt::TensorHandle a3;
  driver_.Execute("tfrt_test.matmul", Location(), matmul_args1,
                  matmul_attrs_ref, a3);

  // This op will finish successfully.
  tfrt::TensorHandle matmul_args2[2] = {a1.CopyRef(), a1.CopyRef()};
  tfrt::TensorHandle a4;
  driver_.Execute("tfrt_test.matmul", Location(), matmul_args2,
                  matmul_attrs_ref, a4);

  tfrt::OpAttrs empty_attrs;
  tfrt::OpAttrsRef empty_attrs_ref = empty_attrs.freeze();
  tfrt::TensorHandle a4_ref = a4.CopyRef();
  driver_.Execute("tfrt_test.print", Location(), a4_ref, empty_attrs_ref, {});

  auto a4_view =
      DHTArrayView<float>(&a4.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a4_view.NumElements(), 1);
  ASSERT_EQ(a4_view.Elements()[0], 4.0);

  // Print the error of a previous op.
  ASSERT_TRUE(a3.GetAsyncTensor()->IsError());
  auto error = a3.GetAsyncTensor()->GetError();
  ASSERT_EQ(error.location->filename, "");
  ASSERT_EQ(error.location->line, -1);

  tfrt::outs() << error << "\n";
  tfrt::outs().flush();
}

void BM_CpuDriverTest(benchmark::State& state) {
  example::CoreRuntimeDriver driver{"cpu"};

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0, 2.0, 2.0, 2.0});
  driver.Execute("tfrt_test.create_dense_tensor",
                 driver.CreateLocation(__FILE__, __LINE__), {}, attrs1.freeze(),
                 a1);

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();

    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a4;
    driver.Execute("tfrt_test.matmul",
                   driver.CreateLocation(__FILE__, __LINE__), matmul_args,
                   matmul_attrs_ref, a4);
  }
}
BENCHMARK(BM_CpuDriverTest);

void BM_CpuMakeOpDriverTest(benchmark::State& state) {
  example::CoreRuntimeDriver driver{"cpu"};

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<ssize_t>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0, 2.0, 2.0, 2.0});
  driver.Execute("tfrt_test.create_dense_tensor",
                 driver.CreateLocation(__FILE__, __LINE__), {}, attrs1.freeze(),
                 a1);

  auto matmul_op = driver.MakeOp("tfrt_test.matmul");

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();

    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a4;
    matmul_op(driver.CreateLocation(__FILE__, __LINE__), matmul_args,
              matmul_attrs_ref, a4, /*chain=*/nullptr);
  }
}
BENCHMARK(BM_CpuMakeOpDriverTest);

}  // namespace
}  // namespace tfrt
