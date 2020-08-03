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

//===- static_registration.cc -----------------------------------*- C++ -*-===//
//
// This file implements the work queue factories and registers them.
//
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <string>
#include <thread>

#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/support/logging.h"

namespace tfrt {

namespace {

static constexpr int kDefaultNumBlockingThreads = 256;

// Factory function for a single-threaded thread pool. The argument must be
// empty.
std::unique_ptr<ConcurrentWorkQueue> SingleThreadedWorkQueueFactory(
    string_view arg) {
  if (!arg.empty()) {
    TFRT_LOG(ERROR) << "Invalid argument for s work queue: "
                    << std::string(arg);
    return nullptr;
  }
  return CreateSingleThreadedWorkQueue();
}

struct MakeMultiThreadedWorkQueue {
  static std::unique_ptr<ConcurrentWorkQueue> make(int num_nonblocking_threads,
                                                   int num_blocking_threads) {
    return CreateMultiThreadedWorkQueue(num_nonblocking_threads,
                                        num_blocking_threads);
  }
};

// Factory function for a multi-threaded thread pool.  Parses the given argument
// to determine the construction parameters.  The argument must be either "X" or
// "X,Y", where X and Y are integers. X will determine the number of threads to
// use for blocking work, and Y will determine the number of threads for
// nonblocking work. If X is not specified, the pool will use a number of
// threads based on the number of CPUs in the system. Y is not specified, a
// `kDefaultNumBlockingThreads` of threads will be used for blocking work.
template <typename MakeWorkQueue>
std::unique_ptr<ConcurrentWorkQueue> MultiThreadedWorkQueueFactory(
    string_view arg) {
  if (arg.empty()) {
    int num_nonblocking = std::thread::hardware_concurrency();
    int num_blocking = kDefaultNumBlockingThreads;
    return MakeWorkQueue::make(num_nonblocking, num_blocking);
  } else {
    size_t comma = arg.find(',');
    int num_threads;
    int num_blocking = kDefaultNumBlockingThreads;
    if (comma == std::string::npos) {
      size_t pos;
      num_threads = std::stoi(std::string(arg), &pos);
      if (pos != arg.size()) {
        TFRT_LOG(ERROR) << "Invalid argument for mstd work queue: "
                        << std::string(arg);
        return nullptr;
      }
    } else {
      size_t pos;
      num_threads =
          std::stoi(std::string(arg.data(), arg.data() + comma), &pos);
      if (pos != comma) {
        TFRT_LOG(ERROR) << "Invalid argument for mstd work queue: "
                        << std::string(arg);
        return nullptr;
      }
      num_blocking = std::stoi(
          std::string(arg.data() + comma + 1, arg.data() + arg.size()), &pos);
      if (pos != arg.size() - comma - 1) {
        TFRT_LOG(ERROR) << "Invalid argument for mstd work queue: "
                        << std::string(arg);
        return nullptr;
      }
    }
    return MakeWorkQueue::make(num_threads, num_blocking);
  }
}

}  // namespace

TFRT_WORK_QUEUE_FACTORY("s", SingleThreadedWorkQueueFactory);
TFRT_WORK_QUEUE_FACTORY(
    "mstd", MultiThreadedWorkQueueFactory<MakeMultiThreadedWorkQueue>);

}  // namespace tfrt
