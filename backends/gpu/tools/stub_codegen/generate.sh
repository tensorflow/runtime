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
# Note: the paths in this script need manual fixing to work in OSS.

set -eux

# Build the tools and generate the HIP header.
bazel build --nocheck_visibility \
  //backends/gpu/tools/stub_codegen:header_codegen \
  //backends/gpu/tools/stub_codegen:impl_codegen

# Generate header and implementation files.
HDR_PATH="third_party/hip/%s_stub.h.inc"
SRC_PATH="third_party/hip/%s_stub.cc.inc"
for API in "hip" "rocblas" "rocfft" "rocsolver" "miopen"; do
   ./bazel-bin/backends/gpu/tools/stub_codegen/header_codegen \
       $(dirname $0)/$API.json | clang-format > $(printf $HDR_PATH $API)
   ./bazel-bin/backends/gpu/tools/stub_codegen/impl_codegen \
       $(dirname $0)/$API.json | clang-format > $(printf $SRC_PATH $API)
done
