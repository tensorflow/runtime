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

//===- test/gpu/test_ops.cc -------------------------------------*- C++ -*-===//
//
// This file defines dispatch functions for GPU test operations.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <limits>
#include <tuple>

#include "llvm_derived/Support/raw_ostream.h"
#include "test_cuda_kernels.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/device/conversion_function.h"
#include "tfrt/gpu/memory/gpu_buffer.h"
#include "tfrt/gpu/ops/test/gpu_ops_and_kernels.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"

namespace tfrt {
using gpu::stream::OwningEvent;

// TODO(tfrt-devs): CoreRT device (corert.executeop) takes TensorHandle
// inputs, and produce TensorHandle outputs. This operation simply passes input
// to output. This op must not be used in production, it's for tests and
// benchmarks only. Figure out how to express side-effectful operations that do
// not need tensors.
static llvm::Expected<gpu::DenseGpuTensor> GpuStreamSynchronize(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& input,
    const TensorMetadata& result_md) {
  if (auto err = gpu::stream::StreamSynchronize(dctx->stream())) {
    return std::move(err);
  }
  return input.CopyRef();
}

static llvm::Expected<std::tuple<TensorMetadata, TensorMetadata>>
ReturnMultipleResultsMD() {
  return std::make_tuple(TensorMetadata(DType(DType::F32), {}),
                         TensorMetadata(DType(DType::F32), {}));
}

static llvm::Expected<std::tuple<gpu::DenseGpuTensor, gpu::DenseGpuTensor>>
ReturnMultipleResults(GpuDispatchContext* dctx,
                      const TensorMetadata& result_md0,
                      const TensorMetadata& result_md1) {
  llvm::Expected<RCReference<gpu::GpuBuffer>> buffer_or_error0 =
      dctx->allocator()->Allocate(
          /*size=*/result_md0.GetHostSizeInBytes(), dctx->stream());
  if (!buffer_or_error0) return buffer_or_error0.takeError();
  RCReference<gpu::GpuBuffer> buffer0 = std::move(*buffer_or_error0);

  llvm::Expected<RCReference<gpu::GpuBuffer>> buffer_or_error1 =
      dctx->allocator()->Allocate(
          /*size=*/result_md1.GetHostSizeInBytes(), dctx->stream());
  if (!buffer_or_error1) return buffer_or_error1.takeError();
  RCReference<gpu::GpuBuffer> buffer1 = std::move(*buffer_or_error1);

  return std::make_tuple(gpu::DenseGpuTensor(result_md0.shape, result_md0.dtype,
                                             std::move(buffer0)),
                         gpu::DenseGpuTensor(result_md1.shape, result_md1.dtype,
                                             std::move(buffer1)));
}

static llvm::Expected<std::tuple<gpu::DenseGpuTensor, gpu::DenseGpuTensor>>
ReturnMultipleResultsWithError(GpuDispatchContext* dctx,
                               const TensorMetadata& result_md0,
                               const TensorMetadata& result_md1) {
  return MakeStringError("error from ReturnMultipleResultsWithError op");
}

static llvm::Expected<gpu::DenseGpuTensor> CreateDenseTensorOp(
    GpuDispatchContext* dctx, const OpAttrsRef& attrs,
    const TensorMetadata& result_md, const ExecutionContext& exec_ctx) {
  size_t size_in_bytes = result_md.GetHostSizeInBytes();
  auto host_buffer = HostBuffer::CreateUninitialized(
      /*size=*/size_in_bytes,
      /*alignment=*/result_md.dtype.GetHostAlignment(),
      exec_ctx.host()->allocator());
  if (!host_buffer) {
    return MakeStringError("Failed to allocate host buffer");
  }

  auto values = attrs.GetRawAsserting("values");
  if (!values.IsArray() || values.array_size == 1) {
    switch (result_md.dtype.kind()) {
      default:
        assert(0 && "invalid result_md dtype");
#define DTYPE_NUMERIC(ENUM)                                            \
  case DType::ENUM: {                                                  \
    using HostType = TypeForDTypeKind<DType::ENUM>;                    \
    auto begin = static_cast<HostType*>(host_buffer->data());          \
    auto end = begin + result_md.shape.GetNumElements();               \
    std::fill(begin, end, *static_cast<const HostType*>(values.data)); \
  } break;
#include "tfrt/dtype/dtype.def"
    }
  } else {
    // TODO(tfrt-devs): Check that data size matches size_in_bytes.
    memcpy(host_buffer->data(), values.GetData(), size_in_bytes);
  }

  DenseHostTensor tensor{result_md, std::move(host_buffer)};
  return gpu::ConvertDenseHostTensorToDenseGpuTensor(
      dctx->current_context(), dctx->stream(), dctx->allocator(), tensor,
      exec_ctx.host());
}

static AsyncValueRef<DenseHostTensor> GpuTensorToHostTensorOp(
    GpuDispatchContext* dctx, const gpu::DenseGpuTensor& input,
    const TensorMetadata& result_md, const ExecutionContext& exec_ctx) {
  return gpu::ConvertDenseGpuTensorToDenseHostTensor(
      dctx->current_context(), dctx->stream(), input, exec_ctx.host());
}

static llvm::Expected<gpu::DenseGpuTensor> DHTToGpuTensorOp(
    GpuDispatchContext* dctx, const DenseHostTensor& input,
    const TensorMetadata& result_md, const ExecutionContext& exec_ctx) {
  return gpu::ConvertDenseHostTensorToDenseGpuTensor(
      dctx->current_context(), dctx->stream(), dctx->allocator(), input,
      exec_ctx.host());
}

static TensorMetadata UnaryIdentityMD(const TensorMetadata& input) {
  return input;
}

static void PrintDhtFullPrecision(const DenseHostTensor& dht) {
  const auto& shape = dht.shape();
  tfrt::outs() << "DenseHostTensor dtype = " << dht.dtype()
               << ", shape = " << shape << ", values = ";

  auto element_size = dht.dtype().GetHostSize();
  auto* data_ptr = static_cast<const char*>(dht.data());

  tfrt::outs() << '[';
  // Print at most 32 elements for a tensor
  static const ssize_t kThreshold = 32;
  for (ssize_t i = 0, e = std::min(kThreshold, dht.NumElements()); i != e;
       ++i) {
    if (i != 0) tfrt::outs() << ", ";
    dht.dtype().PrintFullPrecision(data_ptr + i * element_size, tfrt::outs());
  }

  if (dht.NumElements() > kThreshold) {
    tfrt::outs() << ", ... ";
  }

  tfrt::outs() << ']';
}

// TODO(b/149044322): This op should be a side-effect op which returns a Chain.
// Prints `input` tensor in full precision if it is a DenseHostTensor.
// Else, prints using input.Print().
static void PrintOp(const Tensor& input) {
  const DenseHostTensor* dht = llvm::dyn_cast<const DenseHostTensor>(&input);
  if (dht != nullptr) {
    PrintDhtFullPrecision(*dht);
  } else {
    input.Print(tfrt::outs());
  }
  tfrt::outs() << '\n';
  tfrt::outs().flush();
}

// A simple op for testing OptionalOpArg.
static gpu::DenseGpuTensor TestOptionalArgOp(
    const gpu::DenseGpuTensor& input,
    OptionalOpArg<gpu::DenseGpuTensor> input2) {
  if (input2) {
    return input2->CopyRef();
  } else {
    return input.CopyRef();
  }
}

// A simple op for testing VariadicArgOp.
// TODO(tfrt-devs): Allow variadic output.
static gpu::DenseGpuTensor TestVariadicArgOp(
    const gpu::DenseGpuTensor& input,
    RepeatedArguments<gpu::DenseGpuTensor> input2) {
  if (input2.size() > 0) {
    return input2[0].CopyRef();
  } else {
    return input.CopyRef();
  }
}

void RegisterTestGPUOps(GpuOpRegistry* registry) {
  registry->AddOp("tfrt_test.synchronize", TFRT_GPU_OP(GpuStreamSynchronize));
  registry->AddMetadataFn("tfrt_test.synchronize",
                          TFRT_METADATA(UnaryIdentityMD));

  registry->AddOp("tfrt_test.create_dense_tensor",
                  TFRT_GPU_OP(CreateDenseTensorOp), {"shape", "values"});

  // tfrt_test.gpu_tensor_to_host_tensor is a GPU specific function so the
  // metadata function lives here too.
  registry->AddOp("tfrt_test.gpu_tensor_to_host_tensor",
                  TFRT_GPU_OP(GpuTensorToHostTensorOp));
  registry->AddMetadataFn("tfrt_test.gpu_tensor_to_host_tensor",
                          TFRT_METADATA(UnaryIdentityMD));
  // tfrt_test.dht_to_gpu_tensor is a GPU specific function so the metadata
  // function lives here too.
  registry->AddOp("tfrt_test.dht_to_gpu_tensor", TFRT_GPU_OP(DHTToGpuTensorOp));
  registry->AddMetadataFn("tfrt_test.dht_to_gpu_tensor",
                          TFRT_METADATA(UnaryIdentityMD));

  // TODO(b/149044322): Please don't use this op until we support side-effect
  // op in gpu_device.
  registry->AddOp("tfrt_test.print", TFRT_GPU_OP(PrintOp));

  registry->AddOp("tfrt_test.test_optional_arg",
                  TFRT_GPU_OP(TestOptionalArgOp));
  registry->AddOp("tfrt_test.test_variadic_arg",
                  TFRT_GPU_OP(TestVariadicArgOp));

  // tfrt_test.return_multiple_results op is test-only, and checks that we can
  // return multiple results.
  registry->AddOp("tfrt_test.return_multiple_results",
                  TFRT_GPU_OP(ReturnMultipleResults));
  registry->AddMetadataFn("tfrt_test.return_multiple_results",
                          TFRT_METADATA(ReturnMultipleResultsMD));
  registry->AddOp("tfrt_test.return_multiple_results_with_error",
                  TFRT_GPU_OP(ReturnMultipleResultsWithError));
  registry->AddMetadataFn("tfrt_test.return_multiple_results_with_error",
                          TFRT_METADATA(ReturnMultipleResultsMD));
  RegisterTestCudaKernelsGpuOps(registry);
}
}  // namespace tfrt
