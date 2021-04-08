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

// Declares helpers for tf.Pad op
//
// Declares a function to call tf.Pad implementation on GPU.
#ifndef TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_NONCUDA_H_
#define TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_NONCUDA_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {

class DenseView;
class TensorMetadata;
class OpAttrsRef;
class GpuDispatchContext;

namespace gpu {

class DenseGpuTensor;

llvm::Expected<DenseGpuTensor> CallGpuPadOp(GpuDispatchContext* dctx,
                                            const DenseGpuTensor& input,
                                            const DenseView& paddings,
                                            const TensorMetadata& result_md);

}  // namespace gpu

}  // namespace tfrt

#endif  // TFRT_BACKENDS_GPU_LIB_OPS_TF_PAD_OP_NONCUDA_H_
