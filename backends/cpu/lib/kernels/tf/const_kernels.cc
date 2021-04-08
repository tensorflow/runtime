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

// const Tensorflow kernels.

#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace {

static void ConstKernel(DenseHostTensor* out, DenseAttr attr) {
  assert(CreateTensorMetadata(attr) == out->metadata());
  std::memcpy(out->data(), attr.GetElements(), out->DataSizeInBytes());
}

}  // namespace

namespace tf {
void RegisterConstCpuKernels(KernelRegistry* registry) {
  registry->AddSyncKernel("tf_sync.Const", TFRT_SYNC_KERNEL(ConstKernel));
}
}  // namespace tf

}  // namespace tfrt
