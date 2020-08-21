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

//===- test_native_functions.cc -------------------------------------------===//
//
// This file implements a few native functions for testing.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/native_function.h"
#include "tfrt/test_kernels.h"

namespace tfrt {
namespace {

// A native function that can be used as a sink for any number of arguments.
void NativeSink(AsyncValue* const* arguments, int num_arguments,
                RCReference<AsyncValue>* results, int num_results,
                HostContext* host) {
  assert(num_results == 0);
}

// A native function that can be used as a sink for any number of arguments, and
// returns a single chain.
void NativeAsyncSink(AsyncValue* const* arguments, int num_arguments,
                     RCReference<AsyncValue>* results, int num_results,
                     HostContext* host) {
  assert(num_results == 1);
  results[0] = MakeAvailableAsyncValueRef<Chain>(host);
}

void NativeAdd(AsyncValue* const* arguments, int num_arguments,
               RCReference<AsyncValue>* results, int num_results,
               HostContext* host) {
  assert(num_arguments == 2);
  int32_t a = arguments[0]->get<int32_t>();
  int32_t b = arguments[1]->get<int32_t>();

  assert(num_results == 1);
  results[0] = MakeAvailableAsyncValueRef<int32_t>(host, a + b);
}

void NativeAsyncAdd(AsyncValue* const* arguments, int num_arguments,
                    RCReference<AsyncValue>* results, int num_results,
                    HostContext* host) {
  assert(num_arguments == 2);
  int32_t a = arguments[0]->get<int32_t>();
  int32_t b = arguments[1]->get<int32_t>();

  assert(num_results == 1);
  results[0] = host->EnqueueWork([c = a + b]() { return c; });
}

void NativeError(AsyncValue* const* arguments, int num_arguments,
                 RCReference<AsyncValue>* results, int num_results,
                 HostContext* host) {
  assert(num_results == 1);
  results[0] = MakeErrorAsyncValueRef(host, "something bad happened");
}

}  // namespace

void RegisterTestNativeFunctions(NativeFunctionRegistry* registry) {
  registry->Add("native_sink", NativeSink);
  registry->Add("native_async_sink", NativeAsyncSink);
  registry->Add("native_add", NativeAdd);
  registry->Add("native_async_add", NativeAsyncAdd);
  registry->Add("native_error", NativeError);
}

}  // namespace tfrt
