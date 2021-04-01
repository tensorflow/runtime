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

//===- cudart_stub.cc -------------------------------------------*- C++ -*-===//
//
// Implementation of the CUDA runtime API forwarding calls to symbols
// dynamically loaded from the real library.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "cuda_runtime_api.h"  // from @cuda_headers
#include "symbol_loader.h"

// Memoizes load of the .so for this CUDA library.
static void *LoadSymbol(const char *symbol_name) {
  static SymbolLoader loader("libcudart.so");
  return loader.GetAddressOfSymbol(symbol_name);
}

template <typename Func>
static Func *GetFunctionPointer(const char *symbol_name, Func *func = nullptr) {
  return reinterpret_cast<Func *>(LoadSymbol(symbol_name));
}

// Calls function 'symbol_name' in shared library with 'args'.
// TODO(csigg): Change to 'auto Func' when C++17 is allowed.
template <typename Func, Func *, typename... Args>
static cudaError_t DynamicCall(const char *symbol_name, Args &&...args) {
  static auto func_ptr = GetFunctionPointer<Func>(symbol_name);
  if (!func_ptr) return cudaErrorSharedObjectSymbolNotFound;
  return func_ptr(std::forward<Args>(args)...);
}

#define __dv(x)

extern "C" {
#include "cudart_stub.cc.inc"

const char *CUDARTAPI cudaGetErrorName(cudaError_t error) {
  static auto func_ptr =
      GetFunctionPointer("cudaGetErrorName", cudaGetErrorName);
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(error);
}

const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  static auto func_ptr =
      GetFunctionPointer("cudaGetErrorString", cudaGetErrorString);
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(error);
}

// Following are private symbols in libcudart required for kernel registration.

char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
  static auto func_ptr =
      GetFunctionPointer("__cudaInitModule", __cudaInitModule);
  if (!func_ptr) return 0;
  return func_ptr(fatCubinHandle);
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim, int *wSize) {
  static auto func_ptr =
      GetFunctionPointer("__cudaRegisterFunction", __cudaRegisterFunction);
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
  static auto func_ptr = GetFunctionPointer("__cudaUnregisterFatBinary",
                                            __cudaUnregisterFatBinary);
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                 char *deviceAddress, const char *deviceName,
                                 int ext, size_t size, int constant,
                                 int global) {
  static auto func_ptr =
      GetFunctionPointer("__cudaRegisterVar", __cudaRegisterVar);
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
           constant, global);
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  static auto func_ptr =
      GetFunctionPointer("__cudaRegisterFatBinary", __cudaRegisterFatBinary);
  if (!func_ptr) return nullptr;
  return func_ptr(fatCubin);
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  return DynamicCall<decltype(__cudaPopCallConfiguration),
                     __cudaPopCallConfiguration>(
      "__cudaPopCallConfiguration", gridDim, blockDim, sharedMem, stream);
}

__host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = nullptr) {
  static auto func_ptr = GetFunctionPointer("__cudaPushCallConfiguration",
                                            __cudaPushCallConfiguration);
  if (!func_ptr) return 0;
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  static auto func_ptr = GetFunctionPointer("__cudaRegisterFatBinaryEnd",
                                            __cudaRegisterFatBinaryEnd);
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

}  // extern "C"
