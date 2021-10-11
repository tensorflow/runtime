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

#include "tfrt/utils/mlir_runner_util.h"

#include "gtest/gtest.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/cpp_tests/error_util.h"

namespace tfrt {
namespace testing {
namespace {

TEST(MlirRunnerUtilTest, Sanity) {
  TfrtMlirRunner::Builder builder;
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<compiler::TFRTDialect>();
  context.appendDialectRegistry(registry);
  EXPECT_EQ(&builder.set_mlir_fn_name("test_fn")
                 .set_mlir_input("test_mlir_string")
                 .add_input<int64_t>(1)
                 .add_input<std::string>("abc")
                 .add_input<float>(2.0)
                 .set_mlir_context(&context),
            &builder);
}

TEST(MlirRunnerUtilTest, CompileAndRunSanity) {
  TfrtMlirRunner::Builder builder;
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<compiler::TFRTDialect>();
  context.appendDialectRegistry(registry);
  EXPECT_EQ(&builder.set_mlir_fn_name("main")
                 .set_mlir_input(R"mlir(func @main(%arg0: i32) -> i32 {
                     %x = tfrt.add.i32 %arg0, %arg0
                     tfrt.return %x : i32
                 })mlir")
                 .add_input<int32_t>(42)
                 .set_mlir_context(&context),
            &builder);
  auto runner = builder.Compile();
  auto results = runner.Run();
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->get<int32_t>(), 84);
}

}  // namespace
}  // namespace testing
}  // namespace tfrt
