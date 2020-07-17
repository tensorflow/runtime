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

//===- sync_kernel_test.cc ---------------------------------------*- C++-*-===//
//
// Unit test for TFRT sync kernels.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {

// A constant kernel that takes attribute.
int IntConstant(Attribute<int> arg) { return arg.get(); }

// Integer add kernel.
int IntAdd(int arg0, int arg1) { return arg0 + arg1; }

// Integer divide kernel. Return Error if the divisor is zero.
Expected<int> IntDivide(int arg0, int arg1) {
  if (arg1 == 0) return MakeStringError("Divide by zero");
  return arg0 / arg1;
}

// Return both remainder and quote.
// Return Error if the divisor is zero.
Expected<std::pair<int, int>> IntQuoRem(int arg0, int arg1) {
  if (arg1 == 0) return MakeStringError("Divide by zero");
  return std::make_pair(arg0 / arg1, arg0 % arg1);
}

std::tuple<int, int, int> OneTwoThree() { return std::make_tuple(1, 2, 3); }

void IntCopy(int a, int* b) { *b = a; }

class SyncKernelTest : public ::testing::Test {
 protected:
  HostContext host_context_{[](const DecodedDiagnostic&) {},
                            CreateMallocAllocator(),
                            CreateSingleThreadedWorkQueue()};
  RCReference<RequestContext> req_ctx_ =
      RequestContext::Create(&host_context_, /*resource_context=*/nullptr);
  ExecutionContext exec_ctx_{req_ctx_.CopyRef()};

  SyncKernelFrameBuilder kernel_frame_{exec_ctx_};
};

TEST_F(SyncKernelTest, IntConstant) {
  int attr = 2;
  Value result;
  kernel_frame_.AddAttribute(&attr);
  kernel_frame_.AddResult(&result);

  TFRT_SYNC_KERNEL(IntConstant)(&kernel_frame_);
  auto results = kernel_frame_.GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(result.get<int>(), 2);
}

TEST_F(SyncKernelTest, IntAdd) {
  Value arg1{3};
  Value arg2{2};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddResult(&result);

  TFRT_SYNC_KERNEL(IntAdd)(&kernel_frame_);
  auto results = kernel_frame_.GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(result.get<int>(), 5);
}

TEST_F(SyncKernelTest, IntDivide) {
  Value arg1{3};
  Value arg2{2};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddResult(&result);

  TFRT_SYNC_KERNEL(IntDivide)(&kernel_frame_);
  ASSERT_EQ(result.get<int>(), 1);
}

TEST_F(SyncKernelTest, IntDivideByZero) {
  Value arg1{1};
  Value arg2{0};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddResult(&result);

  TFRT_SYNC_KERNEL(IntDivide)(&kernel_frame_);

  ASSERT_FALSE(result.HasValue());
  ASSERT_TRUE(static_cast<bool>(kernel_frame_.TakeError()));
}

TEST_F(SyncKernelTest, IntQuoRem) {
  Value arg1{5};
  Value arg2{2};
  Value quo;
  Value rem;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddResult(&quo);
  kernel_frame_.AddResult(&rem);

  TFRT_SYNC_KERNEL(IntQuoRem)(&kernel_frame_);

  ASSERT_EQ(quo.get<int>(), 2);
  ASSERT_EQ(rem.get<int>(), 1);
}

TEST_F(SyncKernelTest, TupleResult) {
  Value one, two, three;
  kernel_frame_.AddResult(&one);
  kernel_frame_.AddResult(&two);
  kernel_frame_.AddResult(&three);

  TFRT_SYNC_KERNEL(OneTwoThree)(&kernel_frame_);

  ASSERT_EQ(one.get<int>(), 1);
  ASSERT_EQ(two.get<int>(), 2);
  ASSERT_EQ(three.get<int>(), 3);
}

TEST_F(SyncKernelTest, SideEffectingKernel) {
  Value one{1};
  Value copy{0};
  kernel_frame_.AddArg(&one);
  kernel_frame_.AddArg(&copy);

  TFRT_SYNC_KERNEL(IntCopy)(&kernel_frame_);

  ASSERT_EQ(copy.get<int>(), 1);
}

TEST_F(SyncKernelTest, Reset) {
  Value one, two, three;
  kernel_frame_.AddArg(&one);
  kernel_frame_.AddArg(&two);
  kernel_frame_.AddResult(&three);

  kernel_frame_.Reset();

  ASSERT_EQ(kernel_frame_.GetNumArgs(), 0);
  ASSERT_EQ(kernel_frame_.GetNumResults(), 0);
  ASSERT_EQ(kernel_frame_.GetNumAttributes(), 0);
}

using SyncKernelDeathTest = SyncKernelTest;

// Assert death only in the debug mode, as the validity of kernel call is only
// validated in the debug mode.
#ifndef NDEBUG
#define ASSERT_DEATH_IN_DEBUG(...) ASSERT_DEBUG_DEATH(__VA_ARGS__)
#else
#define ASSERT_DEATH_IN_DEBUG(...)
#endif

TEST_F(SyncKernelDeathTest, TooFewArguments) {
  Value arg1{1};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddResult(&result);

  ASSERT_DEATH_IN_DEBUG((TFRT_SYNC_KERNEL(IntAdd)(&kernel_frame_)), "");
}

TEST_F(SyncKernelDeathTest, TooFewAttributes) {
  Value result;
  kernel_frame_.AddResult(&result);

  ASSERT_DEATH_IN_DEBUG((TFRT_SYNC_KERNEL(IntConstant)(&kernel_frame_)), "");
}

TEST_F(SyncKernelDeathTest, ExtraArguments) {
  Value arg1{1};
  Value arg2{2};
  Value arg3{3};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddArg(&arg3);
  kernel_frame_.AddResult(&result);

  ASSERT_DEATH_IN_DEBUG((TFRT_SYNC_KERNEL(IntAdd)(&kernel_frame_)),
                        "Extra arguments passed to kernel.");
}

TEST_F(SyncKernelDeathTest, ExtraAttributes) {
  Value arg1{1};
  Value arg2{2};
  Value result;
  kernel_frame_.AddArg(&arg1);
  kernel_frame_.AddArg(&arg2);
  kernel_frame_.AddAttribute(nullptr);
  kernel_frame_.AddResult(&result);

  ASSERT_DEATH_IN_DEBUG((TFRT_SYNC_KERNEL(IntAdd)(&kernel_frame_)),
                        "Extra attributes passed to kernel.");
}

}  // namespace
}  // namespace tfrt
