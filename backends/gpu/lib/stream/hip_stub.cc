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

//===- hip_stub.cc ----------------------------------------------*- C++ -*-===//
//
// Implementation of the HIP API forwarding calls to symbols dynamically loaded
// from the real library.
//
//===----------------------------------------------------------------------===//
#include "tfrt/gpu/stream/hip_stub.h"

#include <dlfcn.h>

#include "tfrt/support/logging.h"

static void* LoadSymbol(const char* symbol_name) {
  static void* handle = [&] {
    auto ptr = dlopen("librocm.so", RTLD_LAZY);
    if (!ptr) TFRT_LOG_ERROR << "Failed to load librocm.so";
    return ptr;
  }();
  return handle ? dlsym(handle, symbol_name) : nullptr;
}

#define __dparm(x)
#define DEPRECATED(x)

extern "C" {
#include "hip_stub.cc.inc"
}

// The functions below have a different return type and therefore don't fit
// the code generator patterns.

const char* hipGetErrorName(hipError_t hip_error) {
  using FuncPtr = const char* (*)(hipError_t);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("hipGetErrorName"));
  if (!func_ptr) return nullptr;
  return func_ptr(hip_error);
}

const char* hipGetErrorString(hipError_t hip_error) {
  using FuncPtr = const char* (*)(hipError_t);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("hipGetErrorString"));
  if (!func_ptr) return nullptr;
  return func_ptr(hip_error);
}
