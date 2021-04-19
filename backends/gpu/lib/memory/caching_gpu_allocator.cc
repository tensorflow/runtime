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

//===- caching_gpu_allocator.cc - CUDA CachingGpuAllocator  ---------------===//
//
// This file implements the C++ interface to CUDA caching allocator.

#include "tfrt/gpu/memory/caching_gpu_allocator.h"

#include <cstdint>

#include "llvm/Support/Errc.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

CachingGpuAllocator::CachingGpuAllocator(AsyncValueRef<GpuContext> context)
    : context_(std::move(context)) {}

CachingGpuAllocator::~CachingGpuAllocator() = default;

llvm::Expected<RCReference<gpu::GpuCrtBuffer>> CachingGpuAllocator::Allocate(
    size_t size, wrapper::Stream stream) {
  // FIXME(sanjoy): context handling needs to be cleaned up.  We should not be
  // calling CuStreamGetCtx here.
  auto context = wrapper::CuStreamGetCtx(stream);
  if (!context) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Failed to get context associated with the stream.");
  }

  llvm::Expected<wrapper::CurrentContext> current_context =
      wrapper::CtxGetCurrent();
  if (!current_context) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Could not get a context.");
  }

  assert(current_context->context() == *context);

  auto device_memory = wrapper::MemAlloc(*current_context, size);
  if (!device_memory) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Failed to allocate device memory.");
  }
  // Explicitly release memory to avoid automatic deallocation.
  wrapper::Pointer<void> pointer = device_memory->release();

  allocations_.insert({pointer, context.get()});
  return MakeRef<gpu::GpuCrtBuffer>(pointer, size, this);
}

void CachingGpuAllocator::Deallocate(const gpu::GpuCrtBuffer& buffer) {
  llvm::ExitOnError die_if_error;
  auto it = allocations_.find(buffer.pointer());
  assert(it != allocations_.end());
  // FIXME(sanjoy): context handling needs to be cleaned up.  We should not be
  // calling CuStreamGetCtx here.
  wrapper::CurrentContext current_context =
      die_if_error(wrapper::CtxGetCurrent());
  assert(current_context.context() == it->second);
  die_if_error(wrapper::CtxSynchronize(current_context));
  die_if_error(wrapper::MemFree(it->first));
  allocations_.erase(it);
}

llvm::Error CachingGpuAllocator::RecordUsage(const gpu::GpuCrtBuffer&,
                                             wrapper::Stream) {
  // Since the current simple implementation synchronizes the whole context,
  // we do not care what other streams the buffer was used on.
  return llvm::Error::success();
}

}  // namespace gpu
}  // namespace tfrt
