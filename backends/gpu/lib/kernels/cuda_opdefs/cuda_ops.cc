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

// This file implements MLIR operations for the cuda_ops library.

#include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops.h"

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

CUDADialect::CUDADialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_cuda", context, TypeID::get<CUDADialect>()) {
  context->getOrLoadDialect<tfrt::t::TensorDialect>();
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_opdefs.cpp.inc"
      >();
}

namespace conversion {

CUDA_ConversionDialect::CUDA_ConversionDialect(MLIRContext *context)
    : Dialect(/*name*/ "tfrt_cuda_conversion", context,
              TypeID::get<CUDA_ConversionDialect>()) {
  allowUnknownTypes();
  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_conversion_helper_opdefs.cpp.inc"
      >();
}

mlir::OpFoldResult CastAnyToAnyOp::fold(
    llvm::ArrayRef<mlir::Attribute> operands) {
  // Casting from type A to type B, and then casting back to type A can be
  // folded away.
  mlir::Type output_type = output().getType();

  CastAnyToAnyOp input_op = input().getDefiningOp<CastAnyToAnyOp>();
  if (!input_op) return nullptr;

  mlir::Value indirect_input = input_op.input();
  if (indirect_input.getType() == output_type) return indirect_input;
  return nullptr;
}

}  // namespace conversion

}  // namespace cuda
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_opdefs.cpp.inc"

#define GET_OP_CLASSES
#include "tfrt/gpu/kernels/cuda_opdefs/cuda_conversion_helper_opdefs.cpp.inc"
