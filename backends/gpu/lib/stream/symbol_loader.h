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

// Loads a symbol from a shared library.
#ifndef TFRT_BACKENDS_GPU_LIB_STREAM_SYMBOL_LOADER_H_
#define TFRT_BACKENDS_GPU_LIB_STREAM_SYMBOL_LOADER_H_

#include <string>

#include "llvm/Support/DynamicLibrary.h"
#include "tfrt/support/logging.h"

// Wrapper of llvm::sys::DynamicLibrary.
class SymbolLoader {
 public:
  explicit SymbolLoader(const char* filename)
      : handle_(GetPermanentLibrary(filename)) {}

  void* GetAddressOfSymbol(const char* symbol_name) {
    return handle_.getAddressOfSymbol(symbol_name);
  }

 private:
  static llvm::sys::DynamicLibrary GetPermanentLibrary(const char* filename) {
    std::string error;
    auto handle =
        llvm::sys::DynamicLibrary::getPermanentLibrary(filename, &error);
    if (!handle.isValid())
      TFRT_LOG_ERROR << "Failed to load " << filename << ": " << error;
    return handle;
  }

  llvm::sys::DynamicLibrary handle_;
};

#endif  // TFRT_BACKENDS_GPU_LIB_STREAM_SYMBOL_LOADER_H_
