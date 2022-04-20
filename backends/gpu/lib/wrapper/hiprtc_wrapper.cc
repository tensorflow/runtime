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
#include "tfrt/support/logging.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::raw_ostream& Print(llvm::raw_ostream& os, hiprtcResult result) {
  const char* msg = hiprtcGetErrorString(result);
  if (msg != nullptr) {
    os << "hiprtc Error: (" << msg << ")";
  } else {
    os <<"Unknown hiprtc Error";
  }
  return os;
}

void internal::ProgramDeleter::operator()(hiprtcProgram prog) const{
  LogIfError(HiprtcDestroyProgram(&prog));
}

llvm::Expected<LibraryVersion> HiprtcVersion(){
  int major, minor;
  RETURN_IF_ERROR(hiprtcVersion(&major, &minor));
  return LibraryVersion{major, minor, 0};
}

llvm::Expected<OwningProgram> HiprtcCreateProgram(const char* src){
  hiprtcProgram prog;
  RETURN_IF_ERROR(hiprtcCreateProgram(&prog, src, "", 
		      0, 
		      nullptr,
		      nullptr)); 
  return OwningProgram(prog);
}

llvm::Error HiprtcDestroyProgram(hiprtcProgram* prog){
  return TO_ERROR(hiprtcDestroyProgram(prog));
}

llvm::Error HiprtcCompileProgram(
                                 hiprtcProgram prog,
                                 llvm::ArrayRef<char*> options){
  return TO_ERROR(hiprtcCompileProgram(prog, 0, nullptr));
}

llvm::Error HiprtcAddNameExpression(hiprtcProgram prog,
                                    const char* name_expression){
  return TO_ERROR(hiprtcAddNameExpression(prog, name_expression));
}

llvm::Error HiprtcGetLoweredName(
                                 hiprtcProgram prog,
                                 const char* name_expression,
                                 llvm::ArrayRef<char*> lowered_name){
  return TO_ERROR(hiprtcGetLoweredName(
                                       prog,
                                       name_expression,
                                       const_cast<const char**>(lowered_name.data())));
}


llvm::Expected<size_t> HiprtcGetProgramLogSize(hiprtcProgram prog){
  size_t log_size = 0;
  RETURN_IF_ERROR(hiprtcGetProgramLogSize(prog, &log_size));
  return log_size;
}

llvm::Expected<std::string> HiprtcGetProgramLog(hiprtcProgram prog, size_t log_size){
  std::string log(log_size, '\0');
  RETURN_IF_ERROR(hiprtcGetProgramLog(prog, &log[0]));
  return std::move(log);
}

llvm::Expected<std::vector<char>> HiprtcGetCode(hiprtcProgram prog, size_t code_size){
  std::vector<char> code(code_size + 1, '\0');
  RETURN_IF_ERROR(hiprtcGetCode(prog, code.data()));
  return std::move(code);
}

llvm::Expected<size_t> HiprtcGetCodeSize(hiprtcProgram prog){
  size_t code_size = 0;
  RETURN_IF_ERROR(hiprtcGetCodeSize(prog, &code_size));
  return code_size;
}

} // namespace wrapper
} // namespace gpu
} // namespace tfrt
