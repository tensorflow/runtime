// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- environment_std.h ----------------------------------------*- C++ -*-===//
//
// ThreadingEnvironment defines how to start, join and detatch threads in
// the blocking and non-blocking work queues.
//
// Uses std::thread implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_STD_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_STD_H_

#include <thread>

#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace internal {

class StdThread {
 public:
  explicit StdThread(std::thread thread) : thread_(std::move(thread)) {}
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

struct StdThreadingEnvironment {
  using Thread = ::tfrt::internal::StdThread;

  template <class Function, class... Args>
  std::unique_ptr<Thread> StartThread(string_view name_prefix, Function&& f,
                                      Args&&... args) const {
    return std::make_unique<Thread>(
        std::thread(std::forward<Function>(f), std::forward<Args>(args)...));
  }

  static uint64_t ThisThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }
};

using ThreadingEnvironment = StdThreadingEnvironment;

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_STD_H_
