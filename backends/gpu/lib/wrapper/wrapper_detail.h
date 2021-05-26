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

// Internal header for wrapper implementation.
#ifndef TFRT_BACKENDS_GPU_LIB_STREAM_WRAPPER_DETAIL_H_
#define TFRT_BACKENDS_GPU_LIB_STREAM_WRAPPER_DETAIL_H_

#include <iterator>
#include <mutex>
#include <unordered_map>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/gpu/wrapper/wrapper.h"

#define RETURN_IF_ERROR(expr)         \
  while (auto _result = expr) {       \
    return MakeError(_result, #expr); \
  }

#define TO_ERROR(expr)                           \
  [](auto _result) -> llvm::Error {              \
    if (!_result) return llvm::Error::success(); \
    return MakeError(_result, #expr);            \
  }(expr)

namespace tfrt {
namespace gpu {
namespace wrapper {

// Instantiate this template once for each concrete status code type T.
template <typename T>
llvm::raw_ostream& internal::operator<<(llvm::raw_ostream& os,
                                        const ErrorData<T>& data) {
  using wrapper::operator<<;  // for T.
  os << "'" << data.expr << "': " << data.result;
  if (data.stack_trace) os << ", stack trace:\n" << data.stack_trace;
  return os;
}

template <typename T>
static T* ToCuda(Pointer<T> ptr) {
  return ptr.raw(Platform::CUDA);
}

template <typename T>
static SmallVector<T*, 16> ToCuda(llvm::ArrayRef<Pointer<T>> ptrs) {
  llvm::SmallVector<T*, 16> result;
  result.reserve(ptrs.size());
  llvm::transform(ptrs, std::back_inserter(result),
                  [](Pointer<T> ptr) { return ptr.raw(Platform::CUDA); });
  return result;
}

template <typename T>
static T* ToRocm(Pointer<T> ptr) {
  return ptr.raw(Platform::ROCm);
}

// Per thread context state which is kept in sync with CUDA's and HIP's internal
// state.
extern thread_local struct ContextTls {
  // Current context platform.
  //
  // We expose one active platform context to the user, i.e. the user sees
  // either the current CUDA *or* HIP context.
  Platform platform = Platform::NONE;

  // Current CUDA and HIP contexts. This allows skipping expensive calls to
  // cu/hipCtxSetCurrent() when the requested context matches the current one.
  // Saves about 25ns (55 cycles, from 53.5ns to 28.5ns at the CtxSetCurrent
  // level) per skipped call.
  // TODO(csigg): Remove this optimization. The savings are not worth the risk.
  //
  // These are updated whenever we call any CUDA or HIP API which changes the
  // context. The user may not change the current context behind our back.
  // In debug builds, we check that this doesn't happen.
  CUcontext cuda_ctx = nullptr;
  hipCtx_t hip_ctx = nullptr;

  // Whether it's safe to skip setting the context. CtxSetCurrent() will call
  // the corresponding CUDA or HIP API even if the requested context matches
  // the members above.
  //
  // A primary context can be current, but inactive (which happens when the
  // internal reference count reaches zero after CtxDevicePrimaryRelease() or
  // when the user sets the current context to nullptr). A primary context
  // can be activated with cu/hipCtxSetCurrent(). The members below reflect
  // whether we know that the current context is not an inactive primary. Put
  // differently, if the current context is an inactive primary, the members
  // below are always false. Setting them false is done conservatively whenever
  // the current context could be an inactive primary.
  bool cuda_may_skip_set_ctx = false;
  bool hip_may_skip_set_ctx = false;

#ifndef NDEBUG
  int ref_count = 0;  // CurrentContext instance count.
#endif
} kContextTls;

// Get device of context which may not be current.
llvm::Expected<CUdevice> CuCtxGetDevice(CUcontext context);
llvm::Expected<hipDevice_t> HipCtxGetDevice(hipCtx_t);

// Check that the current context's platform and internal API state matches.
// Report a fatal error otherwise.
//
// Call this function before calling an API which uses the current context.
void CheckCudaContext(CurrentContext current);
void CheckHipContext(CurrentContext current);

// Return an error in debug builds if there is an existing instance of
// CurrentContext in the calling thread. Otherwise return success.
//
// This is called before changing the current context because semantically all
// CurrentContext instances become invalid. The implementation of CurrentContext
// does not hold any state and it would be functionally correct to keep
// instances while the current context changes.
llvm::Error CheckNoCurrentContext();

// Return an instance of CurrentContext.
CurrentContext CreateCurrentContext();

// Consume error by printing a warning.
void LogIfError(llvm::Error&& error);

// Consume error by terminating the program.
void DieIfError(llvm::Error&& error);

// Return error if platform doesn't match expected.
llvm::Error CheckPlatform(Platform platform, Platform expected);

// Return error that platform is invalid.
llvm::Error InvalidPlatform(Platform platform);

// Return error that platform is unsupported.
llvm::Error UnsupportedPlatform(Platform platform);

// Return error that allocating size_bytes failed.
llvm::Error MakeOomError(CurrentContext current, size_t size_bytes);

// Resource types that are being tracked.
enum class ResourceType {
  kStream,
  kEvent,
  kModule,
  kDeviceMemory,
  kHostMemory,
  kRegisteredMemory,
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ResourceType type);

// Map of resources created by the wrapper API. See CheckNoDanglingResources().
class ResourceMap {
  using Map = std::unordered_map<void*, std::pair<ResourceType, Context>>;

  ResourceMap(Map* map, std::mutex* mutex);

 public:
  static ResourceMap Get();

  void NotifyCreated(ResourceType type, void* resource);
  void NotifyDestroyed(void* resource);
  llvm::Error CheckNoneDangling(Context context);

 private:
  Map* map_;
  std::lock_guard<std::mutex> lock_;
};

// Notifies that the 'resource' has been created by the current context.
inline void NotifyResourceCreated(ResourceType type, void* resource) {
#ifndef NDEBUG
  ResourceMap::Get().NotifyCreated(type, resource);
#endif
}

// Notifies that the 'resource' has destroyed.
inline void NotifyResourceDestroyed(void* resource) {
#ifndef NDEBUG
  ResourceMap::Get().NotifyDestroyed(resource);
#endif
}

// Generates an error when 'context' has live resources. The 'context' does not
// need to be valid, this function can be called after destroying it.
inline llvm::Error CheckNoDanglingResources(Context context) {
#ifndef NDEBUG
  return ResourceMap::Get().CheckNoneDangling(context);
#endif
  return llvm::Error::success();
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_STREAM_WRAPPER_DETAIL_H_
