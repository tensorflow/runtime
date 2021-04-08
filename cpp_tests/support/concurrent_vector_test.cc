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

// Unit test for ConcurrentVector

#include "tfrt/support/concurrent_vector.h"

#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

TEST(ConcurrentVectorTest, SingleThreaded) {
  tfrt::ConcurrentVector<int> vec(1);

  constexpr int kCount = 1000;

  for (int i = 0; i < kCount; ++i) {
    ASSERT_EQ(i, vec.emplace_back(i));
  }

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(i, vec[i]);
  }
}

TEST(ConcurrentVectorTest, OneWriterOneReader) {
  tfrt::ConcurrentVector<int> vec(1);

  constexpr int kCount = 1000;

  std::thread writer([&] {
    for (int i = 0; i < kCount; ++i) {
      ASSERT_EQ(i, vec.emplace_back(i));
    }
  });

  std::thread reader([&] {
    for (int i = 0; i < kCount; ++i) {
      while (i >= vec.size())
        ;
      EXPECT_EQ(i, vec[i]);
    }
  });

  writer.join();
  reader.join();
}

TEST(ConcurrentVectorTest, TwoWritersTwoReaders) {
  tfrt::ConcurrentVector<int> vec(1);

  constexpr int kCount = 1000;

  // Each writer stores from 0 to kCount/2 - 1 to the vector.
  auto writer = [&] {
    for (int i = 0; i < kCount / 2; ++i) {
      vec.emplace_back(i);
    }
  };

  std::thread writer1(writer);
  std::thread writer2(writer);

  // Reader reads all the data from the vector and verifies its content.
  auto reader = [&] {
    std::vector<int> stored;
    for (int i = 0; i < kCount; ++i) {
      while (i >= vec.size())
        ;
      stored.emplace_back(vec[i]);
    }
    std::sort(stored.begin(), stored.end());

    for (int i = 0; i < kCount / 2; ++i) {
      ASSERT_EQ(stored[2 * i], i);
      ASSERT_EQ(stored[2 * i + 1], i);
    }
  };

  std::thread reader1(reader);
  std::thread reader2(reader);

  writer1.join();
  writer2.join();

  reader1.join();
  reader2.join();
}

}  // namespace
