//===- unique_any_test.cc -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit test for UniqueAny
//
//===----------------------------------------------------------------------===//

#include "llvm_derived/Support/unique_any.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

class Foo {
 public:
  explicit Foo(int val) : val_(val) {}

  Foo(const Foo& foo) : copy_ctor_called_(true), val_(foo.val_) {}

  Foo& operator=(const Foo& foo) {
    copy_assignment_called_ = true;
    val_ = foo.val_;
    return *this;
  }

  Foo(Foo&& foo) : move_ctor_called_(true), val_(foo.val_) {}

  Foo& operator=(Foo&& other) {
    move_assignment_called_ = true;
    val_ = other.val_;
    return *this;
  }

  int value() const { return val_; }

  bool CopyCtorCalled() const { return copy_ctor_called_; }
  bool CopyAssignCalled() const { return copy_assignment_called_; }
  bool MoveCtorCalled() const { return move_ctor_called_; }
  bool MoveAssignCalled() const { return move_assignment_called_; }

 private:
  bool copy_ctor_called_ = false;
  bool copy_assignment_called_ = false;
  bool move_ctor_called_ = false;
  bool move_assignment_called_ = false;
  int val_;
};

TEST(UniqueAnyTest, Int) {
  int x = 41;
  UniqueAny any = x;
  ASSERT_TRUE(tfrt::any_isa<int>(any));
  int* y = tfrt::any_cast<int>(&any);
  ASSERT_EQ(*y, 41);
}

TEST(UniqueAnyTest, IntRef) {
  int x = 41;
  UniqueAny any = x;
  ASSERT_TRUE(tfrt::any_isa<int>(any));
  int& y = tfrt::any_cast<int&>(any);
  ASSERT_EQ(y, 41);
}

TEST(UniqueAnyTest, Vector) {
  std::vector<int> x{1, 2, 3};
  UniqueAny any = x;
  ASSERT_TRUE(tfrt::any_isa<std::vector<int>>(any));
  std::vector<int>* vec = tfrt::any_cast<std::vector<int>>(&any);
  ASSERT_EQ(*vec, std::vector<int>({1, 2, 3}));
}

TEST(UniqueAnyTest, VectorRef) {
  std::vector<int> x{1, 2, 3};
  UniqueAny any = x;
  ASSERT_TRUE(tfrt::any_isa<std::vector<int>>(any));
  std::vector<int>& vec = tfrt::any_cast<std::vector<int>&>(any);
  vec.push_back(4);
  ASSERT_EQ(vec, std::vector<int>({1, 2, 3, 4}));
}

TEST(UniqueAnyTest, RValueAnyCast) {
  std::vector<int> x{1, 2, 3};
  UniqueAny any = x;
  ASSERT_TRUE(tfrt::any_isa<std::vector<int>>(any));
  std::vector<int> vec = tfrt::any_cast<std::vector<int>>(std::move(any));
  ASSERT_EQ(vec, std::vector<int>({1, 2, 3}));
}

TEST(UniqueAnyTest, VectorFromInitializerList) {
  UniqueAny any(tfrt::in_place_type<std::vector<int>>, {1, 2, 3});
  ASSERT_TRUE(tfrt::any_isa<std::vector<int>>(any));
  std::vector<int>* vec = tfrt::any_cast<std::vector<int>>(&any);
  ASSERT_EQ(*vec, std::vector<int>({1, 2, 3}));
}

TEST(UniqueAnyTest, MakeUniqueAnyVectorFromInitializerList) {
  UniqueAny any = tfrt::make_unique_any<std::vector<int>>({1, 2, 3});
  ASSERT_TRUE(tfrt::any_isa<std::vector<int>>(any));
  std::vector<int>* vec = tfrt::any_cast<std::vector<int>>(&any);
  ASSERT_EQ(*vec, std::vector<int>({1, 2, 3}));
}

TEST(UniqueAnyTest, MoveOnlyType) {
  auto x = std::make_unique<int>(41);
  UniqueAny any = std::move(x);
  ASSERT_TRUE(tfrt::any_isa<std::unique_ptr<int>>(any));
  std::unique_ptr<int>& up = tfrt::any_cast<std::unique_ptr<int>&>(any);
  ASSERT_EQ(*up, 41);
}

TEST(UniqueAnyTest, FooCopyConstructed) {
  Foo foo(41);
  UniqueAny any = foo;
  ASSERT_TRUE(tfrt::any_isa<Foo>(any));
  Foo* fp = tfrt::any_cast<Foo>(&any);
  ASSERT_TRUE(fp->CopyCtorCalled());
  ASSERT_FALSE(fp->CopyAssignCalled());
  ASSERT_FALSE(fp->MoveCtorCalled());
  ASSERT_FALSE(fp->MoveAssignCalled());
  ASSERT_EQ(fp->value(), 41);
}

TEST(UniqueAnyTest, FooMoveConstructed) {
  Foo foo(41);
  UniqueAny any = std::move(foo);
  ASSERT_TRUE(tfrt::any_isa<Foo>(any));
  Foo* fp = tfrt::any_cast<Foo>(&any);
  ASSERT_FALSE(fp->CopyCtorCalled());
  ASSERT_FALSE(fp->CopyAssignCalled());
  ASSERT_TRUE(fp->MoveCtorCalled());
  ASSERT_FALSE(fp->MoveAssignCalled());
  ASSERT_EQ(fp->value(), 41);
}

TEST(UniqueAnyTest, FooInPlaceConstructed) {
  UniqueAny any(tfrt::in_place_type<Foo>, 41);
  ASSERT_TRUE(tfrt::any_isa<Foo>(any));
  Foo* fp = tfrt::any_cast<Foo>(&any);
  ASSERT_FALSE(fp->CopyCtorCalled());
  ASSERT_FALSE(fp->CopyAssignCalled());
  ASSERT_FALSE(fp->MoveCtorCalled());
  ASSERT_FALSE(fp->MoveAssignCalled());
  ASSERT_EQ(fp->value(), 41);
}

TEST(UniqueAnyTest, MakeUniqueAny) {
  UniqueAny any = tfrt::make_unique_any<Foo>(41);
  ASSERT_TRUE(tfrt::any_isa<Foo>(any));
  Foo* fp = tfrt::any_cast<Foo>(&any);
  ASSERT_FALSE(fp->CopyCtorCalled());
  ASSERT_FALSE(fp->CopyAssignCalled());
  ASSERT_FALSE(fp->MoveCtorCalled());
  ASSERT_FALSE(fp->MoveAssignCalled());
  ASSERT_EQ(fp->value(), 41);
}

}  // namespace
}  // namespace tfrt
