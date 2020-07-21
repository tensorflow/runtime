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
#include <dlfcn.h>

#include "cuda_runtime.h"  // from @cuda_headers
#include "tfrt/support/logging.h"

static void *LoadSymbol(const char *symbol_name) {
  static void *handle = [&] {
    auto ptr = dlopen("libcudart.so", RTLD_LAZY);
    if (!ptr) TFRT_LOG_ERROR << "Failed to load libcudart.so";
    return ptr;
  }();
  return handle ? dlsym(handle, symbol_name) : nullptr;
}

#define __dv(x)

extern "C" {
#include "cudart_stub.cc.inc"

const char *CUDARTAPI cudaGetErrorName(cudaError_t error) {
  using FuncPtr = const char *(CUDARTAPI *)(cudaError_t);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("cudaGetErrorName"));
  if (!func_ptr) return nullptr;
  return func_ptr(error);
}

const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  using FuncPtr = const char *(CUDARTAPI *)(cudaError_t);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("cudaGetErrorString"));
  if (!func_ptr) return nullptr;
  return func_ptr(error);
}

// Following are private symbols in libcudart required for kernel registration.

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim, int *wSize) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle, const char *hostFun,
                                    char *deviceFun, const char *deviceName,
                                    int thread_limit, uint3 *tid, uint3 *bid,
                                    dim3 *bDim, dim3 *gDim, int *wSize);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaRegisterFunction"));
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaUnregisterFatBinary"));
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                 char *deviceAddress, const char *deviceName,
                                 int ext, size_t size, int constant,
                                 int global) {
  using FuncPtr = void(CUDARTAPI *)(
      void **fatCubinHandle, char *hostVar, char *deviceAddress,
      const char *deviceName, int ext, size_t size, int constant, int global);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaRegisterVar"));
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
           constant, global);
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  using FuncPtr = void **(CUDARTAPI *)(void *fatCubin);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaRegisterFatBinary"));
  if (!func_ptr) return nullptr;
  return func_ptr(fatCubin);
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  using FuncPtr = cudaError_t(CUDARTAPI *)(dim3 * gridDim, dim3 * blockDim,
                                           size_t * sharedMem, void *stream);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaPopCallConfiguration"));
  if (!func_ptr) return cudaErrorSharedObjectSymbolNotFound;
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

__host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = nullptr) {
  using FuncPtr = unsigned(CUDARTAPI *)(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem, void *stream);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaPushCallConfiguration"));
  if (!func_ptr) return 0;
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("__cudaRegisterFatBinaryEnd"));
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

}  // extern "C"
