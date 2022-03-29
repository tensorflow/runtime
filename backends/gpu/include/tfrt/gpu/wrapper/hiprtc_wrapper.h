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

llvm::Error HiprtcVersion(int* major, int* minor);
llvm::Error HiprtcAddNameExpression(hiprtcProgram prog,
                                    const char* name_expression);
llvm::Error HiprtcCompileProgram(
                                 hiprtcProgram prog,
                                 int numOptions,
                                 const char** options);
llvm::Error HiprtcCreateProgram(
                                hiprtcProgram* prog,
                                const char* src,
                                const char* name,
                                int numberHeaders,
                                const char** headers,
                                const char** includeNames);
llvm::Error HiprtcDestroyProgram(hiprtcProgram* prog);
llvm::Error HiprtcGetLoweredName(
                                 hiprtcProgram prog,
                                 const char* name_expression,
                                 const char** lowered_name);
llvm::Error HiprtcGetProgramLog(hiprtcProgram prog, char* log);
llvm::Error HiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet);
llvm::Error HiprtcGetCode(hiprtcProgram prog, char* code);
llvm::Error HiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet);
} // namespace tfrt
} // namespace gpu
} // namespace wrapper
#endif //TFRT_GPU_WRAPPER_HIPRTC_WRAPPER_H_
