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

// Thin wrapper around the HIP API adding llvm::Error and explicit context.
#include "tfrt/gpu/wrapper/hiprtc_wrapper.h"

#include "llvm/Support/FormatVariadic.h"
#include "tfrt/gpu/wrapper/hiprtc_stub.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& Print(llvm::raw_ostream& os, hiprtcResult result) {
  const char* msg = hiprtcGetErrorString(result);
  if (msg != nullptr) os << "hiprtc Error: (" << msg << ")";
  return os;
}

llvm::Error HiprtcVersion(int* major, int* minor){
  return TO_ERROR(hiprtcVersion(major, minor));
}

llvm::Error HiprtcAddNameExpression(hiprtcProgram prog,
                                    const char* name_expression){
  return TO_ERROR(hiprtcAddNameExpression(prog, name_expression));
}

llvm::Error HiprtcCompileProgram(
                                 hiprtcProgram prog,
                                 int numOptions,
                                 const char** options){
  return TO_ERROR(hiprtcCompileProgram(prog, numOptions, options));
}

llvm::Error HiprtcCreateProgram(
                                hiprtcProgram* prog,
                                const char* src,
                                const char* name,
                                int numberHeaders,
                                const char** headers,
                                const char** includeNames){
  return TO_ERROR(hiprtcCreateProgram(
                                      prog, 
                                      src, 
                                      name, 
                                      numberHeaders, 
                                      headers, 
                                      includeNames));
}

llvm::Error HiprtcDestroyProgram(hiprtcProgram* prog){
  return TO_ERROR(hiprtcDestroyProgram(prog));
}

llvm::Error HiprtcGetLoweredName(
                                 hiprtcProgram prog,
                                 const char* name_expression,
                                 const char** lowered_name){
  return TO_ERROR(hiprtcGetLoweredName(
                                       prog,
                                       name_expression,
                                       lowered_name));
}

llvm::Error HiprtcGetProgramLog(hiprtcProgram prog, char* log){
  return TO_ERROR(hiprtcGetProgramLog(prog, log));
}

llvm::Error HiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet){
  return TO_ERROR(hiprtcGetProgramLogSize(prog, logSizeRet));
}

llvm::Error HiprtcGetCode(hiprtcProgram prog, char* code){
  return TO_ERROR(hiprtcGetCode(prog, code));
}

llvm::Error HiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet){
  return TO_ERROR(hiprtcGetCodeSize(prog, codeSizeRet));
}

} // namespace tfrt
} // namespace gpu
} // namespace wrapper
