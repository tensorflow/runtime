#!/bin/bash
# Copyright 2020 The TensorFlow Runtime Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Generate stub code from HIP headers.
# It's safe to ignore some errors about missing clib includes.

# Build the tool and clang.
bazel build //third_party/tf_runtime/backends/gpu/tools/stub_codegen \
  //third_party/llvm/llvm-project/clang:clang

TOOL_PATH="./bazel-bin/third_party/tf_runtime/backends/gpu/tools/stub_codegen/stub_codegen"

# Generate HIP files
HDR_PATH="third_party/tf_runtime/third_party/hip/%s_stub.h.inc"
SRC_PATH="third_party/tf_runtime/third_party/hip/%s_stub.cc.inc"
for API in "hip"; do
   $TOOL_PATH --api $API --header | clang-format > $(printf $HDR_PATH $API)
   $TOOL_PATH --api $API          | clang-format > $(printf $SRC_PATH $API)
done
