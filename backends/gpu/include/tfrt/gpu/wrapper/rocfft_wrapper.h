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

// Thin wrapper around the rocFFT API that adapts the library for TFRT
// conventions.

#ifndef TFRT_GPU_STREAM_ROCFFT_WRAPPER_H_
#define TFRT_GPU_STREAM_ROCFFT_WRAPPER_H_

#include "tfrt/gpu/wrapper/rocfft_stub.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

struct RocfftErrorData {
  rocfft_status result;
  const char* expr;
  StackTrace stack_trace;
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const RocfftErrorData& data);
// Wraps a rocfft_status into an llvm::ErrorInfo.
using RocfftErrorInfo = TupleErrorInfo<RocfftErrorData>;
rocfft_status GetResult(const RocfftErrorInfo& info);

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct RocfftExecInfoDeleter {
  using pointer = rocfft_execution_info;
  void operator()(rocfft_execution_info handle) const;
};

struct RocfftPlanDeleter {
  using pointer = rocfft_plan;
  void operator()(rocfft_plan plan) const;
};

struct PlanDescriptionDeleter {
  using pointer = rocfft_plan_description;
  void operator()(rocfft_plan_description description) const;
};

}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningRocfftExecInfo =
    internal::OwningResource<internal::RocfftExecInfoDeleter>;
using OwningRocfftPlan = internal::OwningResource<internal::RocfftPlanDeleter>;
using OwningPlanDescription =
    internal::OwningResource<internal::PlanDescriptionDeleter>;

llvm::Error RocfftSetup();
llvm::Error RocfftCleanup();
llvm::Expected<OwningRocfftPlan> RocfftPlanCreate(
    rocfft_result_placement placement, rocfft_transform_type transform_type,
    rocfft_precision precision, size_t dimensions, size_t* lengths,
    size_t number_of_transforms, rocfft_plan_description description);
llvm::Error RocfftExecute(rocfft_plan plan, Pointer<void*> in_buffer,
                          Pointer<void*> out_buffer,
                          rocfft_execution_info handle);
llvm::Error RocfftPlanDestroy(rocfft_plan plan);
llvm::Error RocfftPlanDescriptionSetDataLayout(
    rocfft_plan_description description, rocfft_array_type in_array_type,
    rocfft_array_type out_array_type, size_t* in_offsets, size_t* out_offsets,
    ArrayRef<size_t> in_strides, size_t in_distance,
    ArrayRef<size_t> out_strides, size_t out_distance);
llvm::Expected<std::string> RocfftGetVersionString(char* buf, size_t len);
llvm::Expected<size_t> RocfftPlanGetWorkBufferSize(rocfft_plan plan);
llvm::Error RocfftPlanGetPrint(rocfft_plan plan);
llvm::Expected<OwningPlanDescription> RocfftPlanDescriptionCreate();
llvm::Error RocfftPlanDescriptionDestroy(rocfft_plan_description description);
llvm::Expected<OwningRocfftExecInfo> RocfftExecutionInfoCreate();
llvm::Error RocfftExecutionInfoDestroy(rocfft_execution_info handle);
llvm::Error RocfftExecutionInfoSetWorkBuffer(rocfft_execution_info info,
                                             Pointer<void> work_buffer,
                                             size_t size_in_bytes);
llvm::Error RocfftExecutionInfoSetStream(rocfft_execution_info handle,
                                         void* stream);
}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_STREAM_ROCFFT_WRAPPER_H_
