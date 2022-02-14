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

#include <memory>
#include <utility>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/constraints.h"
#include "tfrt/jitrt/jitrt.h"

namespace tfrt {

using ::tfrt::jitrt::FunctionType;
using ::tfrt::jitrt::MemrefDesc;
using ::tfrt::jitrt::MemrefType;
using ::tfrt::jitrt::OperandConstraint;
using ::tfrt::jitrt::SymbolicShapesResolver;
using ::tfrt::jitrt::Type;
using ::tfrt::jitrt::UnrankedMemrefType;

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

// Create a function type with empty results from the operands shapes.
static FunctionType GetFunctionType(
    llvm::SmallVector<DType> dtypes,
    llvm::SmallVector<llvm::Optional<SymbolicShape>> shapes) {
  llvm::SmallVector<std::unique_ptr<Type>> operands;
  operands.reserve(shapes.size());

  for (auto tuple : llvm::zip(dtypes, shapes)) {
    auto dtype = std::get<0>(tuple);
    auto shape = std::get<1>(tuple);
    if (shape.hasValue()) {
      operands.push_back(std::make_unique<MemrefType>(*shape, dtype));
    } else {
      operands.push_back(std::make_unique<UnrankedMemrefType>(dtype));
    }
  }

  return FunctionType(std::move(operands), {});
}

// Create fake memref operands from the operands shapes.
static llvm::SmallVector<MemrefDesc> GetFakeMemrefs(
    llvm::SmallVector<SymbolicShape> shapes) {
  llvm::SmallVector<MemrefDesc> memrefs;
  memrefs.reserve(shapes.size());

  for (auto& shape : shapes) {
    MemrefDesc desc;
    desc.sizes.insert(desc.sizes.begin(), shape.begin(), shape.end());
    memrefs.push_back(std::move(desc));
  }

  return memrefs;
}

// A helper function to convert initializer list to a list of shapes.
static llvm::SmallVector<SymbolicShape> SymbolicShapes(
    llvm::SmallVector<SymbolicShape> shapes) {
  return shapes;
}

TEST(SymbolicShapeResolverTest, UnrankedInputs) {
  // Operands: tensor<*xf32>, tensor<?xi32>, tensor<?x4xi1>
  auto dtypes = {DType::F32, DType::I32, DType::I1};

  auto type = GetFunctionType(dtypes, {llvm::None,
                                       {{MemrefType::kDynamicSize}},
                                       {{MemrefType::kDynamicSize, 4}}});

  auto constraints = {OperandConstraint::kResolved,
                      OperandConstraint::kResolved,
                      OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100, 100}, {100}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, -2}, {-2}, {-2, 4}}));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100, 101}, {102}, {103, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, -3}, {-4}, {-5, 4}}));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1, 1, 1}, {1}, {1, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1, 1, 1}, {1}, {1, 4}}));
  }

  {  // Known constants converted to a static dimension.
    auto operands = GetFakeMemrefs({{100, 4}, {4}, {1, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {4}, {1, 4}}));
  }
}

TEST(SymbolicShapeResolverTest, DynamicInputShapes) {
  // Operands: tensor<?xf32>, tensor<?xi32>, tensor<?xi1>
  auto dtypes = {DType::F32, DType::I32, DType::I1};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamicSize}},
                                       {{MemrefType::kDynamicSize}},
                                       {{MemrefType::kDynamicSize}}});

  auto constraints = {OperandConstraint::kResolved,
                      OperandConstraint::kResolved,
                      OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100}, {100}, {100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-2}, {-2}}));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100}, {101}, {102}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-3}, {-4}}));
  }

  {  // Two of the three dimensions are the same.
    auto operands = GetFakeMemrefs({{100}, {101}, {100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2}, {-3}, {-2}}));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1}, {1}, {100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1}, {1}, {-2}}));
  }
}

TEST(SymbolicShapeResolverTest, PartialInputShapes) {
  // Operands: tensor<?x4xf32>, tensor<?x8xi32>, tensor<?xi1>
  auto dtypes = {DType::F32, DType::I32, DType::I1};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamicSize, 4}},
                                       {{MemrefType::kDynamicSize, 8}},
                                       {{MemrefType::kDynamicSize}}});

  auto constraints = {OperandConstraint::kResolved,
                      OperandConstraint::kResolved,
                      OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions are the same at runtime.
    auto operands = GetFakeMemrefs({{100, 4}, {100, 8}, {100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-2, 8}, {-2}}));
  }

  {  // All unknown dimensions are unique at runtime.
    auto operands = GetFakeMemrefs({{100, 4}, {101, 8}, {102}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-3, 8}, {-4}}));
  }

  {  // Two of the three dimensions are the same.
    auto operands = GetFakeMemrefs({{100, 4}, {101, 8}, {100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {-3, 8}, {-2}}));
  }

  {  // Ones converted to a static dimension.
    auto operands = GetFakeMemrefs({{1, 4}, {100, 8}, {1}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{1, 4}, {-2, 8}, {1}}));
  }

  {  // Known constants converted to a static dimension.
    auto operands = GetFakeMemrefs({{100, 4}, {8, 8}, {8}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 3);
    EXPECT_EQ(*symbolic, SymbolicShapes({{-2, 4}, {8, 8}, {8}}));
  }
}

TEST(SymbolicShapeResolverTest, ShapeConstrainedInput) {
  // Operands: tensor<*xf32>, tensor<?x4xi32>
  auto dtypes = {DType::F32, DType::I32};

  auto type =
      GetFunctionType(dtypes, {llvm::None, {{MemrefType::kDynamicSize, 4}}});

  auto constraints = {OperandConstraint::kShape, OperandConstraint::kShape};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown materialized as static shapes.
    auto operands = GetFakeMemrefs({{100, 100}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, 100}, {100, 4}}));
  }
}

TEST(SymbolicShapeResolverTest, ShapeConstrainedInputAfterDynamicInput) {
  // Operands: tensor<?x?xf32>, tensor<?x?xi32>
  auto dtypes = {DType::F32, DType::I32};

  auto type = GetFunctionType(
      dtypes, {{{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}}});

  auto constraints = {OperandConstraint::kResolved, OperandConstraint::kShape};

  SymbolicShapesResolver resolver(type, constraints);

  {  // All unknown dimensions materialized as static shapes (first operand
     // resolved based on seen static shapes of the second one).
    auto operands = GetFakeMemrefs({{100, 50}, {100, 50}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, 50}, {100, 50}}));
  }

  {  // Unknown dimension correctly resolved to a symbolic dimension.
    auto operands = GetFakeMemrefs({{100, 50}, {100, 4}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_EQ(symbolic->size(), 2);
    EXPECT_EQ(*symbolic, SymbolicShapes({{100, -2}, {100, 4}}));
  }
}

TEST(SymbolicShapeResolverTest, IncompatibleInput) {
  // Operands: tensor<?x4xi32>
  auto dtypes = {DType::F32};
  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamicSize, 4}}});
  auto constraints = {OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  {  // Operand of a different rank;
    auto operands = GetFakeMemrefs({{100, 100, 100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_TRUE(mlir::failed(symbolic));
  }

  {  // Operand with mismatched static shape.
    auto operands = GetFakeMemrefs({{100, 100}});
    auto symbolic = resolver.Resolve(operands);

    EXPECT_TRUE(mlir::failed(symbolic));
  }
}

// -------------------------------------------------------------------------- //
// Performance benchmarks are below.
// -------------------------------------------------------------------------- //

static void BM_ResolveFullyDynamic(benchmark::State& state) {
  auto dtypes = {DType::F32, DType::I32, DType::I1, DType::F32};

  auto type = GetFunctionType(
      dtypes, {{{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}}});

  auto constraints = {
      OperandConstraint::kResolved, OperandConstraint::kResolved,
      OperandConstraint::kResolved, OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{1, 2}, {3, 4}, {5, 6}, {7, 8}});

  for (auto _ : state) {
    auto symbolic = resolver.Resolve(operands);
    benchmark::DoNotOptimize(symbolic);
  }
}

static void BM_ResolveSameDynamic(benchmark::State& state) {
  auto dtypes = {DType::F32, DType::I32, DType::I1, DType::F32};

  auto type = GetFunctionType(
      dtypes, {{{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}},
               {{MemrefType::kDynamicSize, MemrefType::kDynamicSize}}});

  auto constraints = {
      OperandConstraint::kResolved, OperandConstraint::kResolved,
      OperandConstraint::kResolved, OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{2, 2}, {2, 2}, {2, 2}, {2, 2}});

  for (auto _ : state) {
    auto symbolic = resolver.Resolve(operands);
    benchmark::DoNotOptimize(symbolic);
  }
}

static void BM_ResolveAsStatic(benchmark::State& state) {
  auto dtypes = {DType::F32, DType::I32, DType::I1, DType::F32};

  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamicSize, 4}},
                                       {{MemrefType::kDynamicSize, 8}},
                                       {{MemrefType::kDynamicSize, 16}},
                                       {{MemrefType::kDynamicSize, 32}}});

  auto constraints = {
      OperandConstraint::kResolved, OperandConstraint::kResolved,
      OperandConstraint::kResolved, OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{32, 4}, {16, 8}, {8, 8}, {4, 32}});

  for (auto _ : state) {
    auto symbolic = resolver.Resolve(operands);
    benchmark::DoNotOptimize(symbolic);
  }
}

static void BM_ResolveAsSymbolic(benchmark::State& state) {
  auto dtypes = {DType::F32, DType::I32, DType::I1, DType::F32};

  auto type = GetFunctionType(dtypes, {{{MemrefType::kDynamicSize, 4}},
                                       {{MemrefType::kDynamicSize, 8}},
                                       {{MemrefType::kDynamicSize, 16}},
                                       {{MemrefType::kDynamicSize, 32}}});

  auto constraints = {
      OperandConstraint::kResolved, OperandConstraint::kResolved,
      OperandConstraint::kResolved, OperandConstraint::kResolved};

  SymbolicShapesResolver resolver(type, constraints);

  auto operands = GetFakeMemrefs({{1, 4}, {2, 8}, {3, 8}, {4, 32}});

  for (auto _ : state) {
    auto symbolic = resolver.Resolve(operands);
    benchmark::DoNotOptimize(symbolic);
  }
}

BENCHMARK(BM_ResolveFullyDynamic);
BENCHMARK(BM_ResolveSameDynamic);
BENCHMARK(BM_ResolveAsStatic);
BENCHMARK(BM_ResolveAsSymbolic);

static void HashSymbolicShapes(benchmark::State& state,
                               ArrayRef<SymbolicShape> symbolic_shapes) {
  for (auto _ : state) {
    auto hash = SymbolicShapesResolver::Hash(symbolic_shapes);
    benchmark::DoNotOptimize(hash);
  }
}

static void BM_Hash1x1(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(1, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash1x4(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(4, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash1x8(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(8, {1});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x1(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(1, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x4(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(4, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

static void BM_Hash2x8(benchmark::State& state) {
  llvm::SmallVector<SymbolicShape> symbolic_shapes(8, {1, 2});
  HashSymbolicShapes(state, symbolic_shapes);
}

BENCHMARK(BM_Hash1x1);
BENCHMARK(BM_Hash1x4);
BENCHMARK(BM_Hash1x8);
BENCHMARK(BM_Hash2x1);
BENCHMARK(BM_Hash2x4);
BENCHMARK(BM_Hash2x8);

}  // namespace tfrt
