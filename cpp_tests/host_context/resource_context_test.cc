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

//===- resource_context_test.cc ---------------------------------*- C++ -*-===//
//
// Unit test for TFRT ResourceContext.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/resource_context.h"

#include "gtest/gtest.h"

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

TEST(ResourceContextTest, GetAndCreate) {
  ResourceContext resource_context;
  SomeResource* rc =
      resource_context.CreateResource<SomeResource>("some_name", 41);
  ASSERT_EQ(rc->GetData(), 41);

  SomeResource* rc2 = resource_context.GetResource<SomeResource>("some_name");
  ASSERT_EQ(rc2->GetData(), 41);
}

}  // namespace
}  // namespace tfrt
