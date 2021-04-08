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

// This file implements protobuf-related kernels.

#include "tfrt/cpu/kernels/proto/example.proto.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace proto {

static AsyncValueRef<tfrt::proto::Example> ParseExampleFromBytes(
    Argument<std::string> data, const ExecutionContext& exec_ctx) {
  using ReturnTy = Expected<tfrt::proto::Example>;

  return EnqueueWork(
      exec_ctx, [data = data.ValueRef(), exec_ctx]() -> ReturnTy {
        TFRT_TRACE_SCOPE(Default, "ParseExampleFromBytes");
        tfrt::proto::Example example;
        if (!example.ParseFromString(data.get())) {
          auto diag =
              EmitError(exec_ctx, "failed to parse example.proto from string");
          return MakeStringError(diag.message);
        }
        return example;
      });
}

static AsyncValueRef<std::string> GetBytesFieldFromExample(
    Argument<tfrt::proto::Example> example, Argument<std::string> key,
    const ExecutionContext& exec_ctx) {
  using ReturnTy = Expected<std::string>;

  return EnqueueWork(exec_ctx,
                     [example = example.ValueRef(), key = key.ValueRef(),
                      exec_ctx]() -> ReturnTy {
                       TFRT_TRACE_SCOPE(Default, "GetBytesFieldFromExample");
                       const auto& feature_map = example->features().feature();
                       if (!feature_map.contains(key.get())) {
                         auto diag = EmitError(exec_ctx, "key ", key.get(),
                                               " is not found in the proto");
                         return MakeStringError(diag.message);
                       }
                       const auto& bytes_list =
                           feature_map.at(key.get()).bytes_list();
                       // Assume that each feature is a scalar.
                       assert(bytes_list.value_size() == 1);
                       return bytes_list.value(0);
                     });
}

static llvm::Expected<int64_t> GetInt64FieldFromExample(
    const tfrt::proto::Example& example, const std::string& key) {
  const auto& feature_map = example.features().feature();
  if (!feature_map.contains(key)) {
    return MakeStringError("key ", key, " is not found in the proto");
  }
  const auto& int64_list = feature_map.at(key).int64_list();
  // Assume that each feature is a scalar.
  assert(int64_list.value_size() == 1);

  return int64_list.value(0);
}

// This is the entrypoint to the library.
void RegisterProtoKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.parse_example_from_bytes",
                      TFRT_KERNEL(ParseExampleFromBytes));
  registry->AddKernel("tfrt_test.get_bytes_field_from_example",
                      TFRT_KERNEL(GetBytesFieldFromExample));
  registry->AddKernel("tfrt_test.get_int64_field_from_example",
                      TFRT_KERNEL(GetInt64FieldFromExample));
}

}  // namespace proto
}  // namespace tfrt
