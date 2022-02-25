/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#include <array>
#include <memory>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/jitrt.h"

namespace tfrt {
namespace jitrt {

struct Context {};

static void BM_ReturnValueConverter(benchmark::State& state) {
  Context context;

  auto dims = std::array<int64_t, 4>({1, 1, 1, 1});
  auto type = std::make_unique<MemrefType>(dims, DType::F32);
  auto memref = StridedMemRefType<float, 4>();

  for (auto _ : state) {
    std::array<RCReference<AsyncValue>, 1> storage;
    RemainingResults results(storage);

    ReturnValueConverter<Context> converter(results, context);
    converter.AddConversion(ReturnMemrefAsDenseHostTensor<Context>);
    converter.AddConversion(ReturnAsyncMemrefAsDenseHostTensor<Context>);
    converter.AddConversion(ReturnAsyncToken<Context>);

    auto converted = converter.ReturnValue(0, type.get(), type.get(), &memref);
    if (mlir::failed(converted)) TFRT_LOG(FATAL) << "Failed to convert memref";
  }
}

static void BM_StaticReturnValueConverter(benchmark::State& state) {
  Context context;

  using ReturnToken = ReturnValueConversion<Context, ReturnAsyncToken<Context>>;

  using ReturnAsyncMemref =
      ReturnValueConversion<Context,
                            ReturnAsyncMemrefAsDenseHostTensor<Context>>;

  using ReturnMemref =
      ReturnValueConversion<Context, ReturnMemrefAsDenseHostTensor<Context>>;

  using BenchmarkedReturnValueConverter =
      StaticReturnValueConverter<Context, ReturnToken, ReturnAsyncMemref,
                                 ReturnMemref>;

  auto dims = std::array<int64_t, 4>({1, 1, 1, 1});
  auto type = std::make_unique<MemrefType>(dims, DType::F32);
  auto memref = StridedMemRefType<float, 4>();

  for (auto _ : state) {
    std::array<RCReference<AsyncValue>, 1> storage;
    RemainingResults results(storage);

    BenchmarkedReturnValueConverter converter(results, context);

    auto converted = converter.ReturnValue(0, type.get(), type.get(), &memref);
    if (mlir::failed(converted)) TFRT_LOG(FATAL) << "Failed to convert memref";
  }
}

BENCHMARK(BM_ReturnValueConverter);
BENCHMARK(BM_StaticReturnValueConverter);

}  // namespace jitrt
}  // namespace tfrt
