/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Thin wrapper around the HIP API adding llvm::Error and explicit context.
#ifndef TFRT_GPU_WRAPPER_HIPRTC_WRAPPER_H_
#define TFRT_GPU_WRAPPER_HIPRTC_WRAPPER_H_

#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/hiprtc_stub.h"

namespace tfrt {
namespace gpu{
namespace wrapper {

raw_ostream& Print(raw_ostream& os, hiprtcResult result);

namespace internal{
struct ProgramDeleter{
  using pointer = hiprtcProgram;
  void operator()(hiprtcProgram prog) const;
};
template <typename D>
using OwningResource = std::unique_ptr<typename D::pointer, D>;
} // namespace internal 

using OwningProgram = internal::OwningResource<internal::ProgramDeleter>;

//llvm::Expected<OwningPogram> CreateProgram(const char* str){
//  hiprtcProgram prog;
//  RETURN_IF_ERROR(hiprtcCreate(&prog, str, "", 0, nullptr,nullptr));
//  return OwningProgram(prog);
//}

llvm::Expected<LibraryVersion> HiprtcGetVersion();
llvm::Error HiprtcAddNameExpression(hiprtcProgram prog,
                                    const char* name_expression);
llvm::Error HiprtcCompileProgram(
                                 hiprtcProgram prog,
                                 llvm::ArrayRef<char*> options);
llvm::Expected<OwningProgram> HiprtcCreateProgram(const char* src);
llvm::Error HiprtcDestroyProgram(hiprtcProgram* prog);
llvm::Error HiprtcGetLoweredName(
                                 hiprtcProgram prog,
                                 const char* name_expression,
                                 llvm::ArrayRef<char*> lowered_name);
llvm::Expected<std::string> HiprtcGetProgramLog(hiprtcProgram prog, size_t log_size);
llvm::Expected<size_t> HiprtcGetProgramLogSize(hiprtcProgram prog);
llvm::Expected<std::vector<char>> HiprtcGetCode(hiprtcProgram prog, size_t code_size);
llvm::Expected<size_t> HiprtcGetCodeSize(hiprtcProgram prog);
} // namespace wrapper
} // namespace gpu
} // namespace tfrt
#endif //TFRT_GPU_WRAPPER_HIPRTC_WRAPPER_H_
