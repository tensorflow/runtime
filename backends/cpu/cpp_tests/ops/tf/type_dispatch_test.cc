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

//  Unit tests for type based Tf ops dispatching.

#include "../../../lib/ops/tf/type_dispatch.h"

#include "gtest/gtest.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {
namespace {

TEST(TypeDispatchTest, Simple) {
  using ResultType = std::pair<bool, DType>;

  auto unsupported = [](DType dtype) -> ResultType { return {false, dtype}; };

  auto supported = [](auto type_tag) -> ResultType {
    using T = decltype(type_tag);
    return {true, GetDType<T>()};
  };

  {  // F32 dispatched to supported lambda.
    internal::TypeDispatch<float> dispatch(DType{DType::F32});
    ResultType expected = {true, DType::F32};
    EXPECT_EQ(dispatch(supported, unsupported), expected);
  }

  {  // F64 dispatched to unsupported lambda.
    internal::TypeDispatch<float> dispatch(DType{DType::F64});
    ResultType expected = {false, DType::F64};
    EXPECT_EQ(dispatch(supported, unsupported), expected);
  }

  {  // F64 dispatched to supported lambda.
    internal::TypeDispatch<float, double> dispatch(DType{DType::F64});
    ResultType expected = {true, DType::F64};
    EXPECT_EQ(dispatch(supported, unsupported), expected);
  }
}

}  // namespace
}  // namespace tfrt
