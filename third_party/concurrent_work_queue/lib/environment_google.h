// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//===- environment_google.h -------------------------------------*- C++ -*-===//
//
// ThreadingEnvironment defines how to start, join and detatch threads in
// the blocking and non-blocking work queues.
//
// Uses Google internal Thread implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_GOOGLE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_GOOGLE_H_

#include <thread>  // for std::thread::id

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"
#include "thread/thread.h"

namespace tfrt {
namespace internal {

class GoogleThread : public ::Thread {
 public:
  explicit GoogleThread(string_view name, llvm::unique_function<void()> body)
      : ::Thread(thread::Options().set_joinable(true), name.str()),
        body_(std::move(body)) {
    Start();
  }
  ~GoogleThread() override { Join(); }
  void Run() override { body_(); }

 private:
  llvm::unique_function<void()> body_;
};

struct GoogleThreadingEnvironment {
  using Thread = ::tfrt::internal::GoogleThread;

  template <class Function, class... Args>
  std::unique_ptr<Thread> StartThread(string_view name_prefix, Function&& f,
                                      Args&&... args) const {
    static_assert(sizeof...(Args) == 0, "Callable arguments are not supported");
    llvm::unique_function<void()> body = std::forward<Function>(f);
    return std::make_unique<Thread>(name_prefix, std::move(body));
  }

  static uint64_t ThisThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }
};

using ThreadingEnvironment = GoogleThreadingEnvironment;

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_ENVIRONMENT_GOOGLE_H_
