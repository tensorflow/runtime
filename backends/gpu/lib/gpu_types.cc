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

// Implementation of the types used in the tfrt_cuda dialect.
#include "tfrt/gpu/gpu_types.h"

#include "tfrt/gpu/module_table.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {
GpuContext::GpuContext(wrapper::OwningContext context)
    : context_(std::move(context)) {}

GpuContext::~GpuContext() = default;

Error GpuContext::SetModuleTable(std::unique_ptr<gpu::ModuleTable> table) {
  if (table_) {
    return MakeStringError("Module table already set for context ",
                           context_.get());
  }
  table_ = std::move(table);
  return Error::success();
}

GpuStream::GpuStream(AsyncValueRef<GpuContext> context,
                     wrapper::OwningStream stream)
    : context_(std::move(context)), stream_(std::move(stream)) {}

GpuStream::~GpuStream() = default;

GpuEvent::GpuEvent(AsyncValueRef<GpuContext> context,
                   wrapper::OwningEvent event)
    : context_(std::move(context)), event_(std::move(event)) {}

GpuEvent::~GpuEvent() = default;
}  // namespace gpu
}  // namespace tfrt
