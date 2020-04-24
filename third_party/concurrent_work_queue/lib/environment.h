// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- environment.h --------------------------------------------*- C++ -*-===//
//
// ThreadingEnvironment defines how to start, join and detatch threads in
// the blocking and non-blocking work queues.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_H_

#include <thread>

namespace tfrt {
namespace internal {

struct StdThreadingEnvironment {
  using Thread = std::thread;

  template <class Function, class... Args>
  std::unique_ptr<Thread> StartThread(Function&& f, Args&&... args) const {
    return std::make_unique<Thread>(std::forward<Function>(f),
                                    std::forward<Args>(args)...);
  }

  static void Join(Thread* thread) { thread->join(); }

  static void Detatch(Thread* thread) { thread->detach(); }

  static uint64_t ThisThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }
};

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_H_
