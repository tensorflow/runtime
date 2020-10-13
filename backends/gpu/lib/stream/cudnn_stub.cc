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

//===- cudnn_stub.cc --------------------------------------------*- C++ -*-===//
//
// Implementation of the cuDNN API forwarding calls to symbols dynamically
// loaded from the real library.
//
//===----------------------------------------------------------------------===//
#include <dlfcn.h>

#include "cudnn.h"  // from @cudnn_headers
#include "tfrt/support/logging.h"

static void* LoadSymbol(const char* symbol_name) {
  static void* handle = [&] {
    auto ptr = dlopen("libcudnn.so", RTLD_LAZY);
    if (!ptr) TFRT_LOG_ERROR << "Failed to load libcudnn.so";
    return ptr;
  }();
  return handle ? dlsym(handle, symbol_name) : nullptr;
}

#define CUDNNWINAPI

extern "C" {
#include "cudnn_stub.cc.inc"

const char* CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
  using FuncPtr = const char*(CUDNNWINAPI*)(cudnnStatus_t);
  static auto func_ptr =
      reinterpret_cast<FuncPtr>(LoadSymbol("cudnnGetErrorString"));
  if (!func_ptr) return nullptr;
  return func_ptr(status);
}
}
