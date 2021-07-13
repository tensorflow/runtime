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

// This file defines dispatch functions for GPU implementation of TF ops.

#include "tfrt/gpu/ops/tf/gpu_ops.h"

#include <cstdint>
#include <limits>

#include "tfrt/common/ops/tf/metadata_functions.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
namespace gpu {

void RegisterBinaryGpuTfOps(GpuOpRegistry* registry);
void RegisterDnnGpuTfOps(GpuOpRegistry* registry);
void RegisterMatmulGpuTfOps(GpuOpRegistry* registry);
void RegisterMlirGpuTfOps(GpuOpRegistry* registry);
void RegisterNullaryGpuTfOps(GpuOpRegistry* registry);
void RegisterPadGpuTfOps(GpuOpRegistry* registry);
void RegisterReductionGpuTfOps(GpuOpRegistry* registry);
void RegisterTransposeGpuTfOps(GpuOpRegistry* registry);
void RegisterUnaryGpuTfOps(GpuOpRegistry* registry);

void RegisterTfGpuOps(GpuOpRegistry* registry) {
  for (const std::pair<llvm::StringRef, OpMetadataFn>& md_function :
       GetAllTFMetadataFunctions()) {
    registry->AddMetadataFn(md_function.first, md_function.second);
  }
  RegisterMatmulGpuTfOps(registry);
  RegisterMlirGpuTfOps(registry);
  RegisterNullaryGpuTfOps(registry);
#ifdef TFRT_GPU_CUDA_ENABLED
  RegisterBinaryGpuTfOps(registry);
  RegisterDnnGpuTfOps(registry);
  RegisterPadGpuTfOps(registry);
  RegisterReductionGpuTfOps(registry);
  RegisterTransposeGpuTfOps(registry);
  RegisterUnaryGpuTfOps(registry);
#endif
}
}  // namespace gpu
}  // namespace tfrt
