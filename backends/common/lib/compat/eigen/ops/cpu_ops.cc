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

//===- cpu_ops.cc -----------------------------------------------*- C++ -*-===//
//
// This file defines dispatch functions for CPU implementation of TF ops.
//
//===----------------------------------------------------------------------===//

#include "../kernels/batch_norm.h"
#include "../kernels/conv2d.h"
#include "../kernels/max_pooling.h"
#include "../kernels/zero_padding.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"

namespace tfrt {
namespace compat {

static AsyncValueRef<DenseHostTensor> TfPadOp(
    const DenseHostTensor& input, const DenseHostTensor& padding,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto output = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!output) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating tensor");
  }

  DHTIndexableView<int32_t, 2> padding_view(&padding);
  const auto& padding_shape = padding_view.FixedShape();
  const FixedRankShape<2> expected_padding_shape({4, 2});
  if (padding_shape != expected_padding_shape) {
    return EmitErrorAsync(exec_ctx, "padding shape shoulsd be (4, 2)");
  }

  AsyncValueRef<Chain> chain;
  switch (input.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype for TfPadOp");
      break;
#define DTYPE_NUMERIC(ENUM)                                              \
  case DType::ENUM:                                                      \
    chain = TfPadImpl<EigenTypeForDTypeKind<DType::ENUM>>(               \
        input, padding_view[{1, 0}], padding_view[{1, 1}],               \
        padding_view[{2, 0}], padding_view[{2, 1}], output.getPointer(), \
        exec_ctx);                                                       \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return ForwardValue(output.getValue(), std::move(chain), host);
}

static AsyncValueRef<DenseHostTensor> TfMaxPoolOp(
    const DenseHostTensor& input, const OpAttrsRef& attrs,
    const TensorMetadata& output_md, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto output = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!output) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating tensor");
  }

  auto padding = attrs.GetStringAsserting("padding");
  auto strides = attrs.GetArrayOptional<ssize_t>("strides");
  auto ksize = attrs.GetArrayOptional<ssize_t>("ksize");
  auto data_format = attrs.GetStringOptional("data_format");

  if (strides.size() != 4) {
    return EmitErrorAsync(exec_ctx, "strides should have 4 elements");
  }
  if (ksize.size() != 4) {
    return EmitErrorAsync(exec_ctx, "ksize should have 4 elements");
  }

  if (data_format.hasValue() && data_format.getValue().str() != "NHWC") {
    return EmitErrorAsync(exec_ctx, "only channel last order is supported");
  }

  std::array<ssize_t, 2> strides_t{strides[1], strides[2]};
  std::array<ssize_t, 2> ksize_t{ksize[1], ksize[2]};

  AsyncValueRef<Chain> chain;
  switch (input.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype for TfMaxPoolOp");
      break;
#define DTYPE_FLOAT(ENUM)                                                   \
  case DType::ENUM:                                                         \
    chain = MaxPoolImpl<EigenTypeForDTypeKind<DType::ENUM>>(                \
        input, output.getPointer(), padding, strides_t, ksize_t, exec_ctx); \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return ForwardValue(output.getValue(), std::move(chain), host);
}

static AsyncValueRef<DenseHostTensor> TfConv2DOp(
    const DenseHostTensor& input, const DenseHostTensor& filter,
    const OpAttrsRef& attrs, const TensorMetadata& output_md,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto output = DenseHostTensor::CreateUninitialized(output_md, host);
  if (!output) {
    return EmitErrorAsync(exec_ctx, "out of memory allocating tensor");
  }

  auto padding = attrs.GetStringAsserting("padding");
  auto strides = attrs.GetArrayOptional<ssize_t>("strides");
  auto data_format = attrs.GetStringOptional("data_format");

  if (data_format.hasValue() && data_format.getValue().str() != "NHWC") {
    return EmitErrorAsync(exec_ctx, "only channel last order is supported");
  }

  if (strides.size() != 4) {
    return EmitErrorAsync(exec_ctx, "strides should have 4 elements");
  }
  std::array<ssize_t, 2> strides_t{strides[1], strides[2]};

  AsyncValueRef<Chain> chain;
  using OutputKernel = llvm::Expected<Eigen::NoOpOutputKernel>;
  auto output_kernel = [](Conv2DParams) -> OutputKernel {
    return Eigen::NoOpOutputKernel();
  };

  switch (input.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype for TfConv2DOp");
      break;
#define DTYPE_NUMERIC(ENUM)                                           \
  case DType::ENUM:                                                   \
    chain = internal::Conv2DImpl<EigenTypeForDTypeKind<DType::ENUM>>( \
        input, filter, output.getPointer(), padding, strides_t,       \
        std::move(output_kernel), exec_ctx);                          \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  return ForwardValue(output.getValue(), std::move(chain), host);
}

static std::array<AsyncValueRef<DenseHostTensor>, 6> TfFusedBatchNormV3Op(
    const DenseHostTensor& input, const DenseHostTensor& scale,
    const DenseHostTensor& bias, const DenseHostTensor& mean,
    const DenseHostTensor& variance, const OpAttrsRef& attrs,
    const TensorMetadata& output_md0, const TensorMetadata& output_md1,
    const TensorMetadata& output_md2, const TensorMetadata& output_md3,
    const TensorMetadata& output_md4, const TensorMetadata& output_md5,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  std::array<AsyncValueRef<DenseHostTensor>, 6> results;

  auto result = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);
  for (int i = 0; i < 6; ++i) {
    results[i] = result.CopyRef();
  }

  auto output = DenseHostTensor::CreateUninitialized(output_md0, host);
  if (!output) {
    result.SetError("out of memory allocating tensor");
    return results;
  }

  if (output_md1.IsValid() || output_md2.IsValid() || output_md3.IsValid() ||
      output_md4.IsValid() || output_md5.IsValid()) {
    result.SetError("TfFusedBatchNormV3Op only supports one valid output");
    return results;
  }

  float epsilon;
  if (!attrs.Get("epsilon", &epsilon)) {
    result.SetError("missing epsilon attribute");
    return results;
  }

  auto data_format = attrs.GetStringOptional("data_format");
  if (data_format.hasValue() && data_format.getValue().str() != "NHWC") {
    result.SetError("only channel last order is supported");
    return results;
  }

  AsyncValueRef<Chain> chain;
  switch (input.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx,
                             "unsupported dtype for TfFusedBatchNormV3Op");
      break;
#define DTYPE_FLOAT(ENUM)                                                 \
  case DType::ENUM:                                                       \
    chain = FusedBatchNormV3Impl<EigenTypeForDTypeKind<DType::ENUM>>(     \
        input, scale, bias, mean, variance, output.getPointer(), epsilon, \
        exec_ctx);                                                        \
    break;
#include "tfrt/dtype/dtype.def"  // NOLINT
  }

  result = ForwardValue(output.getValue(), std::move(chain), host);
  for (int i = 0; i < 6; ++i) {
    results[i] = result.CopyRef();
  }
  return results;
}

}  // namespace compat

void RegisterEigenTFOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tf.Pad", TFRT_CPU_OP(compat::TfPadOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tf.MaxPool", TFRT_CPU_OP(compat::TfMaxPoolOp),
                     CpuOpFlags::NoSideEffects,
                     {"padding", "explicit_paddings", "data_format", "strides",
                      "dilations", "ksize"});
  op_registry->AddOp(
      "tf.Conv2D", TFRT_CPU_OP(compat::TfConv2DOp), CpuOpFlags::NoSideEffects,
      {"padding", "explicit_paddings", "data_format", "strides", "dilations"});
  op_registry->AddOp("tf.FusedBatchNormV3",
                     TFRT_CPU_OP(compat::TfFusedBatchNormV3Op),
                     CpuOpFlags::NoSideEffects, {"data_format", "epsilon"});
}

}  // namespace tfrt
