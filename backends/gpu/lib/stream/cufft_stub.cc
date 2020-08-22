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

//===- cufft_stub.cc --------------------------------------------*- C++ -*-===//
//
// Implementation of the cuFFT API forwarding calls to symbols dynamically
// loaded from the real library.
//
//===----------------------------------------------------------------------===//
#include <dlfcn.h>

#include "cufft.h"    // from @cuda_headers
#include "cufftXt.h"  // from @cuda_headers
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"

static void *LoadSymbol(tfrt::string_view symbol_name) {
  static void *handle = [&] {
    auto ptr = dlopen("libcufft.so", RTLD_LAZY);
    if (!ptr) TFRT_LOG_ERROR << "Failed to load libcufft.so";
    return ptr;
  }();
  return handle ? dlsym(handle, symbol_name.data()) : nullptr;
}

extern "C" {

#include "cufft_stub.cc.inc"
#include "cufftxt_stub.cc.inc"
}
