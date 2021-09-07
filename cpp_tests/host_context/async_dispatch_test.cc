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

// This file contains unit tests for tfrt::SingleThreadedWorkQueue

#include "tfrt/host_context/async_dispatch.h"

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(AsyncDispatchTest, Await) {
  AsyncValueRef<int> av = MakeConstructedAsyncValueRef<int>(42);

  std::thread thread{[&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    av.SetStateConcrete();
  }};

  Await(av.CopyRCRef());
  EXPECT_TRUE(av.IsAvailable());
  thread.join();
}

}  // namespace
}  // namespace tfrt
