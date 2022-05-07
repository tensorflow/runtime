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

// Tests related to MapByType
#include "tfrt/support/map_by_type.h"

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

struct T1 {
  int v = 0;
};

struct T2 {
  int v = 1;
  int* destruct_count;
  explicit T2(int* count) : destruct_count(count) {}
  ~T2() { (*destruct_count)++; }
};

struct MapTag {};

struct OptimizedMapTag {};

TEST(MapByTypeTest, Basic) {
  MapByType<MapTag> map;
  EXPECT_FALSE(map.contains<int>());

  auto& i = map.insert(2);
  EXPECT_TRUE(map.contains<int>());
  EXPECT_TRUE(map.contains<const int>());
  EXPECT_EQ(map.get<int>(), 2);
  EXPECT_EQ(map.get<const int>(), 2);

  i = 3;
  EXPECT_EQ(map.get<int>(), 3);
  EXPECT_EQ(map.get<const int>(), 3);

  EXPECT_EQ(*map.getIfExists<int>(), 3);

  EXPECT_EQ(map.getIfExists<float>(), nullptr);

  EXPECT_FALSE(map.contains<T1>());
  auto& t1 = map.emplace<T1>(T1{2});
  EXPECT_EQ(t1.v, 2);
  EXPECT_TRUE(map.contains<T1>());
  EXPECT_EQ(map.get<T1>().v, 2);

  struct A {
    int v = 3;
  };

  EXPECT_FALSE(map.contains<A>());
  A t2_lvalue;
  auto& t2 = map.insert(t2_lvalue);
  EXPECT_EQ(t2.v, 3);
  EXPECT_TRUE(map.contains<A>());
  EXPECT_EQ(map.get<A>().v, 3);
}

TEST(MapByTypeTest, InsertAll) {
  MapByType<MapTag> map;
  map.insert_all(static_cast<int32_t>(1), static_cast<int64_t>(2));
  EXPECT_TRUE(map.contains<int32_t>());
  EXPECT_TRUE(map.contains<int64_t>());
  EXPECT_FALSE(map.contains<float>());

  EXPECT_EQ(map.get<int32_t>(), 1);
  EXPECT_EQ(map.get<int64_t>(), 2);
}

TEST(MapByTypeTest, OptimizedMap) {
  MapByType<OptimizedMapTag> map;
  map.insert_all(static_cast<int32_t>(1), static_cast<int64_t>(2));
  EXPECT_TRUE(map.contains<int32_t>());
  EXPECT_TRUE(map.contains<int64_t>());
  EXPECT_FALSE(map.contains<float>());
  EXPECT_FALSE(map.contains<T1>());

  EXPECT_EQ(map.get<int32_t>(), 1);
  EXPECT_EQ(map.get<int64_t>(), 2);
}

TEST(MapByTypeTest, DestructorCount) {
  int destruct_count = 0;
  {
    MapByType<MapTag> map2;
    {
      auto t2 = std::make_unique<T2>(&destruct_count);
      auto& t2_ref = map2.emplace<std::unique_ptr<T2>>(std::move(t2));
      EXPECT_EQ(t2_ref->v, 1);
    }
    {
      auto first_use = map2.get<std::unique_ptr<T2>>().get();
      EXPECT_EQ(first_use->v, 1);
    }
    {
      auto second_use = map2.get<std::unique_ptr<T2>>().get();
      EXPECT_EQ(second_use->v, 1);
    }
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

TEST(PtrMapByTypeTest, Basic) {
  PtrMapByType<MapTag> map;
  EXPECT_FALSE(map.contains<int32_t>());

  int32_t i32 = 1;
  map.insert(&i32);

  EXPECT_TRUE(map.contains<int32_t>());
  EXPECT_FALSE(map.contains<const int>());
  EXPECT_EQ(*map.get<int32_t>(), 1);

  EXPECT_EQ(map.getIfExists<int64_t>(), nullptr);
  EXPECT_EQ(*map.getIfExists<int32_t>(), 1);
  EXPECT_EQ(map.getIfExists<int32_t>(), &i32);

  const int32_t ci32 = 2;
  map.insert(&ci32);

  EXPECT_TRUE(map.contains<const int32_t>());
  EXPECT_EQ(*map.get<const int32_t>(), 2);
}

// -------------------------------------------------------------------------- //
// Performance benchmarks are below.
// -------------------------------------------------------------------------- //

static void BM_Insert(benchmark::State& state) {
  for (auto _ : state) {
    MapByType<MapTag> map;
    map.insert(static_cast<int32_t>(1));
    map.insert(static_cast<int64_t>(1));
    map.insert(static_cast<float>(1.0));
    map.insert(static_cast<double>(1.0));
    benchmark::DoNotOptimize(map);
  }
}

static void BM_InsertAll(benchmark::State& state) {
  for (auto _ : state) {
    MapByType<MapTag> map;
    map.insert_all(static_cast<int32_t>(1), static_cast<int64_t>(1),
                   static_cast<float>(1.0), static_cast<double>(1.0));
    benchmark::DoNotOptimize(map);
  }
}

static void BM_InsertAndGet(benchmark::State& state) {
  for (auto _ : state) {
    MapByType<MapTag> map;
    map.insert_all(static_cast<int32_t>(1), static_cast<int64_t>(1),
                   static_cast<float>(1.0), static_cast<double>(1.0));
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

static void BM_InsertAndGetOpt(benchmark::State& state) {
  for (auto _ : state) {
    MapByType<OptimizedMapTag> map;
    map.insert_all(static_cast<int32_t>(1), static_cast<int64_t>(1),
                   static_cast<float>(1.0), static_cast<double>(1.0));
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

static void BM_InsertAndGetPtrs(benchmark::State& state) {
  int32_t i32 = 1;
  int64_t i64 = 1;
  float f32 = 1.0;
  double f64 = 1.0;

  for (auto _ : state) {
    PtrMapByType<MapTag> map;
    map.insert_all(&i32, &i64, &f32, &f64);
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

static void BM_InsertAndGetOptPtrs(benchmark::State& state) {
  int32_t i32 = 1;
  int64_t i64 = 1;
  float f32 = 1.0;
  double f64 = 1.0;

  for (auto _ : state) {
    PtrMapByType<OptimizedMapTag> map;
    map.insert_all(&i32, &i64, &f32, &f64);
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.getIfExists<int32_t>());
    benchmark::DoNotOptimize(map.getIfExists<int64_t>());
    benchmark::DoNotOptimize(map.getIfExists<float>());
    benchmark::DoNotOptimize(map.getIfExists<double>());
  }
}

BENCHMARK(BM_Insert);
BENCHMARK(BM_InsertAll);
BENCHMARK(BM_InsertAndGet);
BENCHMARK(BM_InsertAndGetOpt);
BENCHMARK(BM_InsertAndGetPtrs);
BENCHMARK(BM_InsertAndGetOptPtrs);

}  // namespace
}  // namespace tfrt

TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, int32_t);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, int64_t);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, float);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, double);

TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, int32_t);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, int64_t);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, float);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::OptimizedMapTag, double);
