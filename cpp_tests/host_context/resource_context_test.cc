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

// Unit test for TFRT ResourceContext.

#include "tfrt/host_context/resource_context.h"

#include "gtest/gtest.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace {

class SomeResource {
 public:
  explicit SomeResource(int data) : data_(data) {}
  int GetData() const { return data_; }

 private:
  int data_ = 0;
};

TEST(ResourceContextTest, GetOrCreate) {
  ResourceContext resource_context;
  SomeResource* rc =
      resource_context.GetOrCreateResource<SomeResource>("some_name", 41);
  ASSERT_EQ(rc->GetData(), 41);

  SomeResource* rc2 =
      resource_context.GetOrCreateResource<SomeResource>("some_name", 42);
  ASSERT_EQ(rc2->GetData(), 41);
}

TEST(ResourceContextTest, GetOrDie) {
  ResourceContext resource_context;
  SomeResource* rc =
      resource_context.CreateResource<SomeResource>("some_name", 41);
  ASSERT_EQ(rc->GetData(), 41);

  SomeResource* rc2 =
      resource_context.GetResourceOrDie<SomeResource>("some_name");
  ASSERT_EQ(rc2->GetData(), 41);
}

TEST(ResourceContextDeathTest, GetAndDie) {
  EXPECT_DEATH(ResourceContext().GetResourceOrDie<SomeResource>("some_name"),
               "");
}

TEST(ResourceContextTest, GetNotCreated) {
  ResourceContext resource_context;
  Optional<SomeResource*> resource =
      resource_context.GetResource<SomeResource>("some_name");
  EXPECT_EQ(resource.hasValue(), false);
}

TEST(ResourceContextTest, CreateAndGet) {
  ResourceContext resource_context;
  SomeResource* rc =
      resource_context.CreateResource<SomeResource>("some_name", 41);
  ASSERT_EQ(rc->GetData(), 41);

  Optional<SomeResource*> resource =
      resource_context.GetResource<SomeResource>("some_name");
  EXPECT_EQ(resource.hasValue(), true);
  EXPECT_EQ(resource.getValue()->GetData(), 41);
}

TEST(ResourceContextTest, DestructionOrder) {
  static bool parent_destroyed = false;
  struct Parent {
    ~Parent() { parent_destroyed = true; }
  };

  struct Child {
    ~Child() { EXPECT_FALSE(parent_destroyed); }
  };

  ResourceContext resource_context;
  resource_context.CreateResource<Parent>("parent");

  for (int i = 0; i < 100; ++i) {
    resource_context.CreateResource<Child>(StrCat("child", i));
  }
}

}  // namespace
}  // namespace tfrt
