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

// MLIR op types for cuda_ops library
//
// This file declares the types used in the `tfrt_gpu` dialect.

#ifndef TFRT_GPU_GPU_TYPES_H_
#define TFRT_GPU_GPU_TYPES_H_

#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

using GpuFunction = wrapper::Function;

class GpuContext {
 public:
  explicit GpuContext(wrapper::OwningContext context);
  ~GpuContext();

  GpuContext(GpuContext&&) = default;
  GpuContext& operator=(GpuContext&&) = default;

  const wrapper::OwningContext& operator->() const { return context_; }
  wrapper::Context get() const { return context_.get(); }

  Expected<GpuFunction> GetFunction(uint64_t key, string_view data,
                                    string_view name);

 private:
  wrapper::OwningContext context_;
  llvm::DenseMap<uint64_t, std::pair<wrapper::OwningModule, GpuFunction>>
      functions_;
};

class GpuStream {
 public:
  explicit GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream);
  ~GpuStream();

  GpuStream(GpuStream&&) = default;
  GpuStream& operator=(GpuStream&&) = default;

  const wrapper::OwningStream& operator->() const { return stream_; }
  wrapper::Stream get() const { return stream_.get(); }

  wrapper::Context context() const { return context_->get(); }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningStream stream_;
};

class GpuEvent {
 public:
  explicit GpuEvent(AsyncValueRef<GpuContext> context,
                    wrapper::OwningEvent Eventevent);
  ~GpuEvent();

  GpuEvent(GpuEvent&&) = default;
  GpuEvent& operator=(GpuEvent&&) = default;

  const wrapper::OwningEvent& operator->() const { return event_; }
  wrapper::Event get() const { return event_.get(); }

 private:
  AsyncValueRef<GpuContext> context_;
  wrapper::OwningEvent event_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_GPU_TYPES_H_
