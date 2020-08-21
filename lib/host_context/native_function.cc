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

//===- native_function.cc -------------------------------------------------===//
//
// This file implements class NativeFunction.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/native_function.h"

#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

void NativeFunction::Execute(
    const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<RCReference<AsyncValue>> results) const {
  HostContext* host = exec_ctx.host();

  SmallVector<AsyncValue*, 4> unavailable_args;
  for (auto* av : arguments)
    if (!av->IsAvailable()) unavailable_args.push_back(av);

  // Run immediately if all arguments are ready.
  if (unavailable_args.empty()) {
    callable_(arguments.data(), arguments.size(), results.data(),
              results.size(), host);
    return;
  }

  // Otherwise create references to arguments and allocate indirect results for
  // async execution.
  SmallVector<RCReference<AsyncValue>, 4> args;
  args.reserve(arguments.size());
  for (auto* av : arguments) args.push_back(FormRef(av));

  SmallVector<RCReference<IndirectAsyncValue>, 4> indirect_results;
  indirect_results.reserve(results.size());
  for (auto& av_ref : results) {
    indirect_results.push_back(MakeIndirectAsyncValue(host));
    av_ref = indirect_results.back().CopyRef();
  }

  host->RunWhenReady(
      unavailable_args,
      [this, host, args = std::move(args),
       indirect_results = std::move(indirect_results)]() mutable {
        SmallVector<AsyncValue*, 4> arg_avs;
        arg_avs.reserve(args.size());
        for (const auto& arg : args) arg_avs.push_back(arg.get());

        SmallVector<RCReference<AsyncValue>, 4> results;
        results.resize(indirect_results.size());
        callable_(arg_avs.data(), arg_avs.size(), results.data(),
                  results.size(), host);

        for (int i = 0, e = results.size(); i != e; ++i) {
          assert(results[i]);
          indirect_results[i]->ForwardTo(std::move(results[i]));
        }
      });
}

}  // namespace tfrt
