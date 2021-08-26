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

// Unit test for BLAS wrapper (abstraction layer for cuBLAS and rocmBLAS).

#include "tfrt/gpu/wrapper/ccl_wrapper.h"

#include "common.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

TEST_P(Test, CclVersion) {
  auto platform = GetParam();
  TFRT_ASSERT_AND_ASSIGN(int version, CclGetVersion(platform));
  EXPECT_GT(version, 0);
}

TEST_P(Test, CclCommInitRank) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));

  TFRT_ASSERT_AND_ASSIGN(auto id, CclGetUniqueId(platform));
  TFRT_ASSERT_AND_ASSIGN(auto comm, CclCommInitRank(current,
                                                    /*nranks=*/1, id,
                                                    /*rank=*/0));

  TFRT_ASSERT_AND_ASSIGN(int nranks, CclCommCount(comm.get()));
  EXPECT_EQ(nranks, 1);
  TFRT_ASSERT_AND_ASSIGN(int rank, CclCommUserRank(comm.get()));
  EXPECT_EQ(rank, 0);
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
