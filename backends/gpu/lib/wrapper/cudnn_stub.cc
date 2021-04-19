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

// Implementation of the cuDNN API forwarding calls to symbols dynamically
// loaded from the real library.

#include "cudnn.h"  // from @cudnn_headers
#include "symbol_loader.h"

// Memoizes load of the .so for this CUDA library.
static void *LoadSymbol(const char *symbol_name) {
  static SymbolLoader loader("libcudnn.so");
  return loader.GetAddressOfSymbol(symbol_name);
}

template <typename Func>
static Func *GetFunctionPointer(const char *symbol_name, Func *func = nullptr) {
  return reinterpret_cast<Func *>(LoadSymbol(symbol_name));
}

// Calls function 'symbol_name' in shared library with 'args'.
// TODO(csigg): Change to 'auto Func' when C++17 is allowed.
template <typename Func, Func *, typename... Args>
static cudnnStatus_t DynamicCall(const char *symbol_name, Args &&...args) {
  static auto func_ptr = GetFunctionPointer<Func>(symbol_name);
  if (!func_ptr) return CUDNN_STATUS_NOT_INITIALIZED;
  return func_ptr(std::forward<Args>(args)...);
}

#define CUDNNWINAPI

extern "C" {
#include "cudnn_stub.cc.inc"

const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
  static auto func_ptr = reinterpret_cast<decltype(cudnnGetErrorString) *>(
      LoadSymbol("cudnnGetErrorString"));
  if (!func_ptr) return "FAILED_TO_LOAD_FUNCTION_SYMBOL";
  return func_ptr(status);
}
}  // extern "C"
