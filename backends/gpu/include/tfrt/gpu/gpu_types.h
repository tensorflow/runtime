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
// This file declares the types used in the `tfrt_cuda` dialect.

#ifndef TFRT_GPU_GPU_TYPES_H_
#define TFRT_GPU_GPU_TYPES_H_

#include <memory>

#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {
class ModuleTable;

class GpuContext {
 public:
  explicit GpuContext(gpu::stream::OwningContext context);
  ~GpuContext();

  GpuContext(GpuContext&&) = default;
  GpuContext& operator=(GpuContext&&) = default;

  const gpu::stream::OwningContext& operator->() const { return context_; }
  gpu::stream::Context get() const { return context_.get(); }

  Error SetModuleTable(std::unique_ptr<gpu::ModuleTable> table);
  const gpu::ModuleTable* GetModuleTable() const { return table_.get(); }

 private:
  gpu::stream::OwningContext context_;
  std::unique_ptr<gpu::ModuleTable> table_;
};

class GpuStream {
 public:
  explicit GpuStream(AsyncValueRef<GpuContext> context,
                     gpu::stream::OwningStream stream);
  ~GpuStream();

  GpuStream(GpuStream&&) = default;
  GpuStream& operator=(GpuStream&&) = default;

  const gpu::stream::OwningStream& operator->() const { return stream_; }
  gpu::stream::Stream get() const { return stream_.get(); }

 private:
  AsyncValueRef<GpuContext> context_;
  gpu::stream::OwningStream stream_;
};

class GpuEvent {
 public:
  explicit GpuEvent(AsyncValueRef<GpuContext> context,
                    gpu::stream::OwningEvent Eventevent);
  ~GpuEvent();

  GpuEvent(GpuEvent&&) = default;
  GpuEvent& operator=(GpuEvent&&) = default;

  const gpu::stream::OwningEvent& operator->() const { return event_; }
  gpu::stream::Event get() const { return event_.get(); }

 private:
  AsyncValueRef<GpuContext> context_;
  gpu::stream::OwningEvent event_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_GPU_TYPES_H_
