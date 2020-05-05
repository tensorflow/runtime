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

//===- tf/cpu_ops.cc --------------------------------------------*- C++ -*-===//
//
// This file defines dispatch functions for CPU implementation of TF ops.
//
//===----------------------------------------------------------------------===//

#include "tfrt/cpu/ops/tf/cpu_ops.h"

#include "../../kernels/cpu_kernels.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/ops/tf/metadata_functions.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {
namespace {

//===----------------------------------------------------------------------===//
// tf.Const op
//===----------------------------------------------------------------------===//

static Expected<DenseHostTensor> TfConstOp(const OpAttrsRef& attrs,
                                           const TensorMetadata& dest_md,
                                           HostContext* host) {
  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    return MakeStringError("out of memory allocating dht tensor");
  }

  auto& dest_tensor = dest_alloc.getValue();

  // Copy data from `value` attribute to dht.
  DenseAttr dense_attr = attrs.GetAsserting<DenseAttr>("value");
  std::memcpy(dest_tensor.data(), dense_attr.GetElements(),
              dest_md.GetHostSizeInBytes());

  return std::move(dest_tensor);
}

//===----------------------------------------------------------------------===//
// tf.Add op
//===----------------------------------------------------------------------===//

static AsyncValueRef<HostTensor> TfAddOp(const HostTensor& lhs,
                                         const HostTensor& rhs,
                                         HostContext* host) {
  switch (lhs.dtype().kind()) {
    default:
      assert(0 && "shape function failure");
      return {};
#define DTYPE_NUMERIC(ENUM) \
  case DType::ENUM:         \
    return cpu::Add<EigenTypeForDTypeKind<DType::ENUM>>(lhs, rhs, host);
#include "tfrt/tensor/dtype.def"
  }
}

//===----------------------------------------------------------------------===//
// tf.Matmul op
//===----------------------------------------------------------------------===//

static void TfMatMulOp(const DenseHostTensor& lhs, const DenseHostTensor& rhs,
                       const OpAttrsRef& attrs, const TensorMetadata& dest_md,
                       RCReference<AsyncValue>* dest,
                       const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();

  // Handle attributes.
  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");

  // Computes C = A @ B.
  switch (lhs.dtype().kind()) {
    default:
      *dest = EmitErrorAsync(exec_ctx, "unsupported dtype for matmul");
      return;
#define DTYPE_NUMERIC(ENUM)                                    \
  case DType::ENUM:                                            \
    cpu::CallMatMulKernel<EigenTypeForDTypeKind<DType::ENUM>>( \
        lhs, rhs, &dest_tensor, transpose_a, transpose_b);     \
    break;
#include "tfrt/tensor/dtype.def"  // NOLINT
  }

  *dest =
      host->MakeAvailableAsyncValueRef<DenseHostTensor>(std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// tf.Relu op
//===----------------------------------------------------------------------===//

static AsyncValueRef<DenseHostTensor> TfReluOp(
    const DenseHostTensor& A, const TensorMetadata& B_md,
    const ExecutionContext& exec_ctx) {
  auto* host = exec_ctx.host();

  auto dest = DenseHostTensor::CreateUninitialized(B_md, host);
  if (!dest) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating result");
  }

  AsyncValueRef<Chain> chain;
  switch (A.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype for relu");
      break;
#define DTYPE_NUMERIC(ENUM)                                \
  case DType::ENUM:                                        \
    chain = cpu::Relu<EigenTypeForDTypeKind<DType::ENUM>>( \
        A, dest.getPointer(), exec_ctx);                   \
    break;
#include "tfrt/tensor/dtype.def"  // NOLINT
  }

  auto B_tensor = host->MakeUnconstructedAsyncValueRef<DenseHostTensor>();
  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen([dest = std::move(dest).getValue(),
                     chain = std::move(chain),
                     B_tensor = B_tensor.CopyRef()]() mutable {
    if (chain.IsError()) {
      B_tensor.SetError(chain.GetError());
    } else {
      B_tensor.emplace<DenseHostTensor>(std::move(dest));
    }
  });
  return B_tensor;
}

}  // namespace

void RegisterTfCpuOps(CpuOpRegistry* op_registry) {
  for (const std::pair<llvm::StringRef, OpMetadataFn>& md_function :
       GetAllTFMetadataFunctions()) {
    op_registry->AddMetadataFn(md_function.first, md_function.second);
  }
  op_registry->AddOp("tf.Const", TFRT_CPU_OP(TfConstOp),
                     CpuOpFlags::NoSideEffects, {"value"});
  op_registry->AddOp("tf.AddV2", TFRT_CPU_OP(TfAddOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
  op_registry->AddOp("tf.MatMul", TFRT_CPU_OP(TfMatMulOp),
                     CpuOpFlags::NoSideEffects, {"transpose_a", "transpose_b"});
  op_registry->AddOp("tf.Relu", TFRT_CPU_OP(TfReluOp),
                     CpuOpFlags::NoSideEffects);
}

}  // namespace tfrt
