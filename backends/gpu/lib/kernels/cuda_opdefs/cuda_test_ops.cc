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

// This file implements MLIR operations for the cuda_test_ops library.

#include "tfrt/gpu/kernels/cuda_opdefs/cuda_test_ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/tensor/opdefs/tensor.h"

namespace tfrt {
namespace cuda {

//===----------------------------------------------------------------------===//
// CUDADialect Dialect
//===----------------------------------------------------------------------===//

CUDATestDialect::CUDATestDialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_cuda_test", context,
              TypeID::get<CUDATestDialect>()) {
  context->getOrLoadDialect<tfrt::t::TensorDialect>();
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_test_opdefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_test_opdefs.cpp.inc"

}  // namespace cuda
}  // namespace tfrt
