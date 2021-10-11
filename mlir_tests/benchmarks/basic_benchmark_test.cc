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

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/utils/mlir_runner_util.h"

namespace tfrt {
namespace testing {
namespace {

// The tests in this file correspond to the tests in
// tf_runtime/mlir_tests/bef_executor/benchmark.mlir
void BM_basic_benchmark_with_input(benchmark::State& state) {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<compiler::TFRTDialect>();
  context.appendDialectRegistry(registry);
  TfrtMlirRunner::Builder builder;
  EXPECT_EQ(&builder.set_mlir_fn_name("main")
                 .set_mlir_input(R"mlir(func @main(%arg0: i32) -> i32 {
                     %x = tfrt.add.i32 %arg0, %arg0
                     tfrt.return %x : i32
                 })mlir")
                 .add_input<int32_t>(42)
                 .set_mlir_context(&context),
            &builder);
  auto runner = builder.Compile();

  for (auto _ : state) {
    runner.Run();
  }
}
BENCHMARK(BM_basic_benchmark_with_input);

void BM_basic_benchmark_without_input(benchmark::State& state) {
  TfrtMlirRunner::Builder builder;
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<compiler::TFRTDialect>();
  context.appendDialectRegistry(registry);
  EXPECT_EQ(&builder.set_mlir_fn_name("main")
                 .set_mlir_input(
                     R"mlir(func @main() -> i32 {
                     %c = tfrt.constant.i32 42
                     %x = tfrt.add.i32 %c, %c
                     tfrt.return %x : i32
                 })mlir")
                 .set_mlir_context(&context),
            &builder);
  auto runner = builder.Compile();

  for (auto _ : state) {
    runner.Run();
  }
}
BENCHMARK(BM_basic_benchmark_without_input);

}  // namespace
}  // namespace testing
}  // namespace tfrt
