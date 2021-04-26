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

// Thin wrapper around the rocFFT API adding llvm::Error.
#include "tfrt/gpu/wrapper/rocfft_wrapper.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

template void internal::LogResult(llvm::raw_ostream&, rocfft_status);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocfft_status status) {
  switch (status) {
    case rocfft_status_success:
      return os << "rocfft_status_success";
    case rocfft_status_failure:
      return os << "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:
      return os << "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:
      return os << "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:
      return os << "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides:
      return os << "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:
      return os << "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset:
      return os << "rocfft_status_invalid_offset";
    default:
      return os << llvm::formatv("rocfft_status({0})",
                                 static_cast<int>(status));
  }
}

void internal::RocfftExecInfoDeleter::operator()(
    rocfft_execution_info handle) const {
  LogIfError(RocfftExecutionInfoDestroy(handle));
}

void internal::RocfftPlanDeleter::operator()(rocfft_plan plan) const {
  LogIfError(RocfftPlanDestroy(plan));
}

void internal::PlanDescriptionDeleter::operator()(
    rocfft_plan_description description) const {
  LogIfError(RocfftPlanDescriptionDestroy(description));
}

// TOD(gkg): Figure out format of the version string, parse it and return as
// struct RocfftLibraryVersion {int major; int minor; int patch; };
// to be consistent with CufftGetVersion.
llvm::Expected<std::string> RocfftGetVersionString() {
  char buf[256];  // Minimum 30 characters per rocFFT API
  RETURN_IF_ERROR(rocfft_get_version_string(buf, sizeof(buf)));
  return std::string(buf);
}

llvm::Error RocfftSetup() { return TO_ERROR(rocfft_setup()); }

llvm::Error RocfftCleanup() { return TO_ERROR(rocfft_cleanup()); }

llvm::Expected<OwningPlanDescription> RocfftPlanDescriptionCreate() {
  rocfft_plan_description description;
  RETURN_IF_ERROR(rocfft_plan_description_create(&description));
  return OwningPlanDescription(description);
}

llvm::Error RocfftPlanDescriptionDestroy(rocfft_plan_description description) {
  return TO_ERROR(rocfft_plan_description_destroy(description));
}

llvm::Error RocfftPlanDescriptionSetDataLayout(
    rocfft_plan_description description, rocfft_array_type in_array_type,
    rocfft_array_type out_array_type, size_t* in_offsets, size_t* out_offsets,
    ArrayRef<size_t> in_strides, size_t in_distance,
    ArrayRef<size_t> out_strides, size_t out_distance) {
  return TO_ERROR(rocfft_plan_description_set_data_layout(
      description, in_array_type, out_array_type, in_offsets, out_offsets,
      in_strides.size(), in_strides.data(), in_distance, out_strides.size(),
      out_strides.data(), out_distance));
}

llvm::Expected<OwningRocfftPlan> RocfftPlanCreate(
    rocfft_result_placement placement, rocfft_transform_type transform_type,
    rocfft_precision precision, size_t dimensions, size_t* lengths,
    size_t number_of_transforms, rocfft_plan_description description) {
  rocfft_plan plan;
  RETURN_IF_ERROR(rocfft_plan_create(&plan, placement, transform_type,
                                     precision, dimensions, lengths,
                                     number_of_transforms, description));
  return OwningRocfftPlan(plan);
}

llvm::Error RocfftPlanDestroy(rocfft_plan plan) {
  return TO_ERROR(rocfft_plan_destroy(plan));
}

llvm::Expected<size_t> RocfftPlanGetWorkBufferSize(rocfft_plan plan) {
  size_t size_in_bytes;
  RETURN_IF_ERROR(rocfft_plan_get_work_buffer_size(plan, &size_in_bytes));
  return size_in_bytes;
}

llvm::Error RocfftPlanGetPrint(rocfft_plan plan) {
  return TO_ERROR(rocfft_plan_get_print(plan));
}

llvm::Expected<OwningRocfftExecInfo> RocfftExecutionInfoCreate() {
  rocfft_execution_info handle = nullptr;
  RETURN_IF_ERROR(rocfft_execution_info_create(&handle));
  return OwningRocfftExecInfo(handle);
}

llvm::Error RocfftExecutionInfoDestroy(rocfft_execution_info handle) {
  return TO_ERROR(rocfft_execution_info_destroy(handle));
}

llvm::Error RocfftExecutionInfoSetStream(rocfft_execution_info handle,
                                         Stream stream) {
  return TO_ERROR(rocfft_execution_info_set_stream(
      handle, static_cast<hipStream_t>(stream)));
}

llvm::Error RocfftExecutionInfoSetWorkBuffer(rocfft_execution_info info,
                                             Pointer<void> work_buffer,
                                             size_t size_in_bytes) {
  return TO_ERROR(rocfft_execution_info_set_work_buffer(
      info, ToRocm(work_buffer), size_in_bytes));
}

llvm::Error RocfftExecute(rocfft_plan plan, Pointer<const void> in_buffer,
                          Pointer<void> out_buffer,
                          rocfft_execution_info handle) {
  auto rocm_in_buffer = const_cast<void*>(ToRocm(in_buffer));
  auto rocm_out_buffer = ToRocm(out_buffer);
  return TO_ERROR(
      rocfft_execute(plan, &rocm_in_buffer, &rocm_out_buffer, handle));
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
