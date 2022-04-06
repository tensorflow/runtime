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
#include "tfrt/bef/kernel.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(KernelTest, ResultAndUsers) {
  bef::Buffer buffer;
  bef::Allocator allocator(&buffer);

  auto ctor = bef::New<bef::ResultAndUsers>(&allocator);
  ctor.set_result(100);

  auto users_ctor = ctor.construct_users(/*num_users=*/2);
  users_ctor.ConstructAt(0, 200);
  users_ctor.ConstructAt(1, 300);

  bef::ResultAndUsers view(buffer.Get(ctor.address()));
  EXPECT_EQ(view.result(), 100);

  bef::Vector<uint32_t> users = view.users();
  EXPECT_THAT(users, testing::ElementsAreArray({200, 300}));
}

TEST(KernelTest, Kernel) {
  bef::Buffer buffer;
  bef::Allocator allocator(&buffer);

  bef::Kernel::Constructor ctor = bef::New<bef::Kernel>(&allocator);

  ctor.set_code(100);
  ctor.set_location(200);

  ctor.construct_arguments(/*size=*/2).Assign({400, 500});
  ctor.construct_attributes(/*size=*/1).Assign({1400});
  ctor.construct_functions(/*size=*/0);

  auto results_ctor = ctor.construct_results(/*size=*/2);

  for (uint32_t i = 0; i < 2; ++i) {
    bef::ResultAndUsers::Constructor ru_ctor = results_ctor.ConstructAt(i);
    ru_ctor.set_result(i);
    ru_ctor.construct_users(/*size=*/2).Assign({100 + i, 200 + i});
  }

  bef::Kernel view(buffer.Get(ctor.address()));

  EXPECT_EQ(view.code(), 100);
  EXPECT_EQ(view.location(), 200);

  EXPECT_THAT(view.arguments(), testing::ElementsAreArray({400, 500}));
  EXPECT_THAT(view.attributes(), testing::ElementsAreArray({1400}));
  EXPECT_THAT(view.functions(), testing::IsEmpty());

  bef::Vector<bef::ResultAndUsers> results = view.results();
  ASSERT_EQ(results.size(), 2);

  EXPECT_EQ(results[0].result(), 0);
  EXPECT_THAT(results[0].users(), testing::ElementsAreArray({100, 200}));
  EXPECT_EQ(results[1].result(), 1);
  EXPECT_THAT(results[1].users(), testing::ElementsAreArray({101, 201}));
}

}  // namespace
}  // namespace tfrt
