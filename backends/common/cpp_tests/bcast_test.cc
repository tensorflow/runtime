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

//===- bcast_test.cc --------------------------------------------*- C++ -*-===//
//
// Unit test TF broadcasting rules.
//
//===----------------------------------------------------------------------===//

#include "tfrt/common/ops/tf/bcast.h"

#include <sstream>

#include "gtest/gtest.h"

namespace tfrt {

namespace {
std::string ToString(ArrayRef<ssize_t> dims) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "[";
  llvm::interleaveComma(dims, os);
  os << "]";
  return str;
}

std::string ReshapeOf(Expected<ArgumentBCast>& bcast) {
  if (static_cast<bool>(bcast)) return ToString(bcast->reshape());
  return "<invalid bcast>";
}

std::string BroadcastOf(Expected<ArgumentBCast>& bcast) {
  if (static_cast<bool>(bcast)) return ToString(bcast->broadcast());
  return "<invalid bcast>";
}

std::string ShapeOf(Expected<TensorShape>& shape) {
  if (static_cast<bool>(shape)) {
    SmallVector<ssize_t, 4> dims;
    shape->GetDimensions(&dims);
    return ToString(dims);
  }
  return "<invalid shape>";
}

}  // namespace

TEST(BCastTest, GetBroadcastedShape) {
  {
    auto shape = GetBroadcastedShape(TensorShape({3, 1}), TensorShape({1, 3}));
    ASSERT_EQ(ShapeOf(shape), "[3, 3]");
  }
  {
    auto shape =
        GetBroadcastedShape(TensorShape({3, 1, 1}), TensorShape({1, 4, 2}));
    ASSERT_EQ(ShapeOf(shape), "[3, 4, 2]");
  }
}

TEST(BCastTest, GetArgumentBCast) {
  {
    auto bcast = GetArgumentBCast(TensorShape({3, 1}), TensorShape({3, 3}));
    ASSERT_EQ(ReshapeOf(bcast), "[3, 1]");
    ASSERT_EQ(BroadcastOf(bcast), "[1, 3]");
  }

  {
    auto bcast = GetArgumentBCast(TensorShape(ArrayRef<ssize_t>{3}),
                                  TensorShape({2, 3}));
    ASSERT_EQ(ReshapeOf(bcast), "[1, 3]");
    ASSERT_EQ(BroadcastOf(bcast), "[2, 1]");
  }

  {
    auto bcast = GetArgumentBCast(TensorShape(ArrayRef<ssize_t>{1}),
                                  TensorShape({2, 3}));
    ASSERT_EQ(ReshapeOf(bcast), "[1, 1]");
    ASSERT_EQ(BroadcastOf(bcast), "[2, 3]");
  }

  {
    auto bcast = GetArgumentBCast(TensorShape({3, 3}),
                                  TensorShape(ArrayRef<ssize_t>{3}));
    ASSERT_FALSE(static_cast<bool>(bcast));
  }

  {
    auto bcast = GetArgumentBCast(TensorShape({3, 3}), TensorShape({5, 5}));
    ASSERT_FALSE(static_cast<bool>(bcast));
  }
}

}  // namespace tfrt
