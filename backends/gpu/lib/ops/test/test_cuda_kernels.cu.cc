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

// This file contains CUDA kernels for some test.* ops.

#include "test_cuda_kernels.h"

#include <string>

#define EIGEN_USE_GPU

#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/error_util.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h"  // from @eigen_archive

namespace tfrt {
namespace {
typedef cudaStream_t gpuStream_t;
typedef cudaDeviceProp gpuDeviceProp_t;

template <typename T>
using AlignedEigenVector =
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, int32_t>,
                     Eigen::Aligned>;

template <typename T>
AlignedEigenVector<T> FlattenDenseGpuTensor(const gpu::DenseGpuTensor& dgt) {
  gpu::stream::Pointer<void> pointer = dgt.buffer().pointer();
  return AlignedEigenVector<T>(
      reinterpret_cast<T*>(pointer.raw(pointer.platform())), dgt.NumElements());
}

template <typename T>
AlignedEigenVector<T> FlattenDenseGpuTensor(gpu::DenseGpuTensor* dgt) {
  gpu::stream::Pointer<void> pointer = dgt->buffer().pointer();
  return AlignedEigenVector<T>(
      reinterpret_cast<T*>(pointer.raw(pointer.platform())),
      dgt->NumElements());
}

template <typename T>
void AddTensorsGeneric(Eigen::GpuDevice& device,
                       const gpu::DenseGpuTensor& lhs_tensor,
                       const gpu::DenseGpuTensor& rhs_tensor,
                       gpu::DenseGpuTensor* result_tensor) {
  AlignedEigenVector<T> lhs = FlattenDenseGpuTensor<T>(lhs_tensor);
  AlignedEigenVector<T> rhs = FlattenDenseGpuTensor<T>(rhs_tensor);
  AlignedEigenVector<T> result = FlattenDenseGpuTensor<T>(result_tensor);

  result.device(device) = lhs + rhs;
}

void AddTensors(Eigen::GpuDevice& device, const gpu::DenseGpuTensor& lhs_tensor,
                const gpu::DenseGpuTensor& rhs_tensor,
                gpu::DenseGpuTensor* result_tensor) {
  switch (lhs_tensor.dtype().kind()) {
    default:
      assert(0 && "shape function failure");
#define DTYPE_NUMERIC(ENUM)                                \
  case DType::ENUM:                                        \
    AddTensorsGeneric<EigenTypeForDTypeKind<DType::ENUM>>( \
        device, lhs_tensor, rhs_tensor, result_tensor);    \
    return;
#include "tfrt/dtype/dtype.def"
  }
}

Expected<gpu::DenseGpuTensor> GpuAddOp(GpuDispatchContext* dctx,
                                       const gpu::DenseGpuTensor& tensor_a,
                                       const gpu::DenseGpuTensor& tensor_b,
                                       const OpAttrsRef& attrs,
                                       const TensorMetadata& result_md) {
  size_t size_in_bytes = result_md.GetHostSizeInBytes();

  TFRT_ASSIGN_OR_RETURN(RCReference<gpu::GpuBuffer> buffer,
                        dctx->allocator()->Allocate(
                            /*size=*/size_in_bytes, dctx->stream()));

  auto result =
      gpu::DenseGpuTensor(result_md.shape, result_md.dtype, std::move(buffer));

  AddTensors(*dctx->eigen_gpu_device(), tensor_a, tensor_b, &result);

  return std::move(result);
}
}  // namespace

void RegisterTestCudaKernelsGpuOps(GpuOpRegistry* registry) {
  registry->AddOp("tfrt_test.add", TFRT_GPU_OP(GpuAddOp));
}
}  // namespace tfrt
