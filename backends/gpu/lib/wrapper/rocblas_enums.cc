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

// cuBLAS enum parsers and printers.
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return os << "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return os << "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return os << "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return os << "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return os << "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return os << "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return os << "rocblas_status_internal_error";
    case rocblas_status_perf_degraded:
      return os << "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch:
      return os << "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased:
      return os << "rocblas_status_size_increased";
    case rocblas_status_size_unchanged:
      return os << "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value:
      return os << "rocblas_status_invalid_value";
    case rocblas_status_continue:
      return os << "rocblas_status_continue";
    case rocblas_status_check_numerics_fail:
      return os << "rocblas_status_check_numerics_fail";
    default:
      return os << llvm::formatv("rocblas_status({0})",
                                 static_cast<int>(status));
  }
}

template <>
Expected<rocblas_operation> Parse<rocblas_operation>(llvm::StringRef name) {
  if (name == "rocblas_operation_none") return rocblas_operation_none;
  if (name == "rocblas_operation_transpose") return rocblas_operation_transpose;
  if (name == "rocblas_operation_conjugate_transpose")
    return rocblas_operation_conjugate_transpose;
  return MakeStringError("Unknown rocblas_operation: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_operation value) {
  switch (value) {
    case rocblas_operation_none:
      return os << "rocblas_operation_none";
    case rocblas_operation_transpose:
      return os << "rocblas_operation_transpose";
    case rocblas_operation_conjugate_transpose:
      return os << "rocblas_operation_conjugate_transpose";
    default:
      return os << llvm::formatv("rocblas_operation({0})",
                                 static_cast<int>(value));
  }
}

template <>
Expected<rocblas_datatype> Parse<rocblas_datatype>(llvm::StringRef name) {
  if (name == "rocblas_datatype_f16_r") return rocblas_datatype_f16_r;
  if (name == "rocblas_datatype_f32_r") return rocblas_datatype_f32_r;
  if (name == "rocblas_datatype_f64_r") return rocblas_datatype_f64_r;
  if (name == "rocblas_datatype_f16_c") return rocblas_datatype_f16_c;
  if (name == "rocblas_datatype_f32_c") return rocblas_datatype_f32_c;
  if (name == "rocblas_datatype_f64_c") return rocblas_datatype_f64_c;
  if (name == "rocblas_datatype_i8_r") return rocblas_datatype_i8_r;
  if (name == "rocblas_datatype_u8_r") return rocblas_datatype_u8_r;
  if (name == "rocblas_datatype_i32_r") return rocblas_datatype_i32_r;
  if (name == "rocblas_datatype_u32_r") return rocblas_datatype_u32_r;
  if (name == "rocblas_datatype_i8_c") return rocblas_datatype_i8_c;
  if (name == "rocblas_datatype_u8_c") return rocblas_datatype_u8_c;
  if (name == "rocblas_datatype_i32_c") return rocblas_datatype_i32_c;
  if (name == "rocblas_datatype_u32_c") return rocblas_datatype_u32_c;
  if (name == "rocblas_datatype_bf16_r") return rocblas_datatype_bf16_r;
  if (name == "rocblas_datatype_bf16_c") return rocblas_datatype_bf16_c;
  return MakeStringError("Unknown rocblas_datatype: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_datatype value) {
  switch (value) {
    case rocblas_datatype_f16_r:
      return os << "rocblas_datatype_f16_r";
    case rocblas_datatype_f32_r:
      return os << "rocblas_datatype_f32_r";
    case rocblas_datatype_f64_r:
      return os << "rocblas_datatype_f64_r";
    case rocblas_datatype_f16_c:
      return os << "rocblas_datatype_f16_c";
    case rocblas_datatype_f32_c:
      return os << "rocblas_datatype_f32_c";
    case rocblas_datatype_f64_c:
      return os << "rocblas_datatype_f64_c";
    case rocblas_datatype_i8_r:
      return os << "rocblas_datatype_i8_r";
    case rocblas_datatype_u8_r:
      return os << "rocblas_datatype_u8_r";
    case rocblas_datatype_i32_r:
      return os << "rocblas_datatype_i32_r";
    case rocblas_datatype_u32_r:
      return os << "rocblas_datatype_u32_r";
    case rocblas_datatype_i8_c:
      return os << "rocblas_datatype_i8_c";
    case rocblas_datatype_u8_c:
      return os << "rocblas_datatype_u8_c";
    case rocblas_datatype_i32_c:
      return os << "rocblas_datatype_i32_c";
    case rocblas_datatype_u32_c:
      return os << "rocblas_datatype_u32_c";
    case rocblas_datatype_bf16_r:
      return os << "rocblas_datatype_bf16_r";
    case rocblas_datatype_bf16_c:
      return os << "rocblas_datatype_bf16_c";
    default:
      return os << llvm::formatv("rocblas_datatype({0})",
                                 static_cast<int>(value));
  }
}

template <>
Expected<rocblas_gemm_algo> Parse<rocblas_gemm_algo>(llvm::StringRef name) {
  if (name == "rocblas_gemm_algo_standard") return rocblas_gemm_algo_standard;
  return MakeStringError("Unknown rocblas_gemm_algo: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_gemm_algo value) {
  switch (value) {
    case rocblas_gemm_algo_standard:
      return os << "rocblas_gemm_algo_standard";
    default:
      return os << llvm::formatv("rocblas_gemm_algo({0})",
                                 static_cast<int>(value));
  }
}

template <>
Expected<rocblas_fill> Parse<rocblas_fill>(llvm::StringRef name) {
  if (name == "rocblas_fill_upper") return rocblas_fill_upper;
  if (name == "rocblas_fill_lower") return rocblas_fill_lower;
  if (name == "rocblas_fill_full") return rocblas_fill_full;
  return MakeStringError("Unknown rocblas_fill: ", name);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, rocblas_fill value) {
  switch (value) {
    case rocblas_fill_upper:
      return os << "rocblas_fill_upper";
    case rocblas_fill_lower:
      return os << "rocblas_fill_lower";
    case rocblas_fill_full:
      return os << "rocblas_fill_full";
    default:
      return os << llvm::formatv("rocblas_fill({0})", static_cast<int>(value));
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
