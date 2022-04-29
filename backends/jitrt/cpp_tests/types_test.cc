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

#include "tfrt/jitrt/types.h"

#include <utility>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {
namespace jitrt {

// -------------------------------------------------------------------------- //
// Benchmarks for constructing MemrefDesc.
// -------------------------------------------------------------------------- //

static void BM_CreateMemrefDesc_1d(benchmark::State& state) {
  void* ptr = reinterpret_cast<void*>(0xDEADBEEF);
  int64_t size = 123;
  int64_t stride = 456;

  int64_t num_memrefs = state.range(0);

  for (auto _ : state) {
    llvm::SmallVector<jitrt::MemrefDesc> memrefs;
    memrefs.resize(num_memrefs);

    for (unsigned i = 0; i < num_memrefs; ++i) {
      jitrt::MemrefDesc& memref = memrefs[i];
      memref.dtype = tfrt::DType::I8;
      memref.data = ptr;
      memref.offset = 0;
      memref.sizes.resize_for_overwrite(1);
      memref.sizes[0] = size;
      memref.strides.resize_for_overwrite(1);
      memref.strides[0] = stride;
    }

    benchmark::DoNotOptimize(memrefs);
  }
}

BENCHMARK(BM_CreateMemrefDesc_1d)->Arg(1)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

// -------------------------------------------------------------------------- //
// Run benchmarks for verifying operands.
// -------------------------------------------------------------------------- //

static MemrefDesc GetFakeMemref(ArrayRef<int64_t> sizes) {
  MemrefDesc memref;
  memref.dtype = DType::F32;
  memref.sizes.assign(sizes.begin(), sizes.end());
  return memref;
}

static void BenchmarkVerifyMemrefOperand(benchmark::State& state,
                                         const MemrefDesc& memref) {
  auto sizes = memref.sizes;
  auto dtype = memref.dtype;

  for (auto _ : state) {
    if (auto err = VerifyMemrefOperand(0, dtype, {sizes}, memref)) break;
  }
}

static void BM_VerifyMemref_1d(benchmark::State& state) {
  auto memref = GetFakeMemref({1});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_2d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_3d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_4d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3, 4});
  BenchmarkVerifyMemrefOperand(state, memref);
}

static void BM_VerifyMemref_5d(benchmark::State& state) {
  auto memref = GetFakeMemref({1, 2, 3, 4, 5});
  BenchmarkVerifyMemrefOperand(state, memref);
}

BENCHMARK(BM_VerifyMemref_1d);
BENCHMARK(BM_VerifyMemref_2d);
BENCHMARK(BM_VerifyMemref_3d);
BENCHMARK(BM_VerifyMemref_4d);
BENCHMARK(BM_VerifyMemref_5d);

}  // namespace jitrt
}  // namespace tfrt
