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

// Implementation of the HIP API forwarding calls to symbols dynamically loaded
// from the real library.
#include "tfrt/gpu/wrapper/hip_stub.h"

#include "symbol_loader.h"

// Memoizes load of the .so for this ROCm library.
static void *LoadSymbol(const char *symbol_name) {
  static SymbolLoader loader("libamdhip64.so");
  return loader.GetAddressOfSymbol(symbol_name);
}

template <typename Func>
static Func *GetFunctionPointer(const char *symbol_name, Func *func = nullptr) {
  return reinterpret_cast<Func *>(LoadSymbol(symbol_name));
}

// Calls function 'symbol_name' in shared library with 'args'.
// TODO(csigg): Change to 'auto Func' when C++17 is allowed.
template <typename Func, Func *, typename... Args>
static hipError_t DynamicCall(const char *symbol_name, Args &&...args) {
  static auto func_ptr = GetFunctionPointer<Func>(symbol_name);
  if (!func_ptr) return hipErrorSharedObjectInitFailed;
  return func_ptr(std::forward<Args>(args)...);
}

#define __dparm(x)
#define DEPRECATED(x) [[deprecated]]
#define dim3 hipDim3_t

extern "C" {
#include "hip_stub.cc.inc"
}

// The functions below have a different return type and therefore don't fit
// the code generator patterns.

const char *hipGetErrorName(hipError_t hip_error) {
  static auto func_ptr = GetFunctionPointer("hipGetErrorName", hipGetErrorName);
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(hip_error);
}

const char *hipGetErrorString(hipError_t hip_error) {
  static auto func_ptr =
      GetFunctionPointer("hipGetErrorString", hipGetErrorString);
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(hip_error);
}

const char *hiprtcGetErrorString(hiprtcResult result) {
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetErrorString", hiprtcGetErrorString);
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(result);
}

hiprtcResult hiprtcVersion(int* major, int* minor){
  static auto func_ptr =
      GetFunctionPointer("hiprtcVersion", hiprtcVersion);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(major, minor);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name_expression){
  static auto func_ptr =
      GetFunctionPointer("hiprtcAddNameExpression", hiprtcAddNameExpression);
  if (!func_ptr) return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
  return func_ptr(prog, name_expression);
}

hiprtcResult hiprtcCompileProgram(
                                   hiprtcProgram prog,
                                   int numOptions,
                                   const char** options){
  static auto func_ptr =
      GetFunctionPointer("hiprtcCompileProgram", hiprtcCompileProgram);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, numOptions, options);
}

hiprtcResult hiprtcCreateProgram(
                                  hiprtcProgram* prog,
                                  const char* src,
                                  const char* name,
                                  int numberHeaders,
                                  char** headers,
                                  const char** includeNames){
  static auto func_ptr =
      GetFunctionPointer("hiprtcCreateProgram", hiprtcCreateProgram);
  if (!func_ptr) return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE;
  return func_ptr(prog, src, name, numberHeaders, headers, includeNames);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog){
  static auto func_ptr =
      GetFunctionPointer("hiprtcDestroyProgram", hiprtcDestroyProgram);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog);
}

hiprtcResult hiprtcGetLoweredName(
                                  hiprtcProgram prog,
                                  const char* name_expression,
                                  const char** lowered_name){
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetLoweredName", hiprtcGetLoweredName);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, name_expression, lowered_name);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log){
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetProgramLog", hiprtcGetProgramLog);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, log);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet){
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetProgramLogSize", hiprtcGetProgramLogSize);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, logSizeRet);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code){
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetCode", hiprtcGetCode);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, code);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet){
  static auto func_ptr =
      GetFunctionPointer("hiprtcGetCodeSize", hiprtcGetCodeSize);
  if (!func_ptr) return HIPRTC_ERROR_INTERNAL_ERROR;
  return func_ptr(prog, codeSizeRet);
}
