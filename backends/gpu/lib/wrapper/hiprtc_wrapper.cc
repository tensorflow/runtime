// Copyright 2022 The TensorFlow Runtime Authors
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

// Thin wrapper around the HIP API adding llvm::Error and explicit context.
#include "tfrt/gpu/wrapper/hiprtc_wrapper.h"

#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tfrt/gpu/wrapper/hiprtc_stub.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& Print(llvm::raw_ostream& os, hiprtcResult result) {
  if (const char* msg = hiprtcGetErrorString(result)) return os << msg;
  switch (result) {
    case HIPRTC_SUCCESS:
      return os << "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return os << "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return os << "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return os << "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return os << "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return os << "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return os << "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return os << "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return os << "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return os << "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return os << "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return os << "HIPRTC_ERROR_INTERNAL_ERROR";
    default:
      return os << llvm::formatv("hiprtcResult({0})", static_cast<int>(result));
  }
}

void internal::ProgramDeleter::operator()(hiprtcProgram program) const {
  LogIfError(HiprtcDestroyProgram(&program));
}

llvm::Expected<LibraryVersion> HiprtcVersion() {
  int major, minor;
  RETURN_IF_ERROR(hiprtcVersion(&major, &minor));
  return LibraryVersion{major, minor, 0};
}

llvm::Expected<OwningProgram> HiprtcCreateProgram(const char* src) {
  hiprtcProgram program;
  RETURN_IF_ERROR(hiprtcCreateProgram(&program, src, /*name=*/"",
                                      /*numHeaders=*/0, /*headers=*/nullptr,
                                      /*headerNames=*/nullptr));
  return OwningProgram(program);
}

llvm::Error HiprtcDestroyProgram(hiprtcProgram* program) {
  return TO_ERROR(hiprtcDestroyProgram(program));
}

llvm::Error HiprtcCompileProgram(hiprtcProgram program,
                                 llvm::ArrayRef<const char*> options) {
  return TO_ERROR(hiprtcCompileProgram(
      program, options.size(), const_cast<const char**>(options.data())));
}

llvm::Error HiprtcAddNameExpression(hiprtcProgram program,
                                    const char* name_expression) {
  return TO_ERROR(hiprtcAddNameExpression(program, name_expression));
}

llvm::Error HiprtcGetLoweredName(hiprtcProgram program,
                                 const char* name_expression,
                                 llvm::ArrayRef<char*> lowered_name) {
  return TO_ERROR(hiprtcGetLoweredName(
      program, name_expression, const_cast<const char**>(lowered_name.data())));
}

llvm::Expected<std::string> HiprtcGetProgramLog(hiprtcProgram program) {
  size_t log_size = 0;
  RETURN_IF_ERROR(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, '\0');
  RETURN_IF_ERROR(hiprtcGetProgramLog(program, &log.front()));
  return log;
}

llvm::Expected<std::string> HiprtcGetCode(hiprtcProgram program) {
  size_t code_size = 0;
  RETURN_IF_ERROR(hiprtcGetCodeSize(program, &code_size));
  std::string code(code_size, '\0');
  RETURN_IF_ERROR(hiprtcGetCode(program, &code.front()));
  return code;
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
