// Copyright 2021 The TensorFlow Runtime Authors
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

// RUN: tfrt_gpu_opt \
// RUN:   -test-set-entry-point='platform=CUDA buffer_sizes=1,2,3' %s \
// RUN: | FileCheck %s

// CHECK: func @get_tfrt_gpu_entry_point() -> !tfrt_gpu.entry_point {
// CHECK: %[[result:.*]] = "tfrt_gpu.get_entry_point"() {
// CHECK-SAME:  buffer_sizes = [1, 2, 3],
// CHECK-SAME:  function_name = "main",
// CHECK-SAME:  platform = 1 : si32,
// CHECK-SAME:  version = {{[[:digit:]]+}} : i64
// CHECK-SAME: } : () -> !tfrt_gpu.entry_point
// CHECK: tfrt.return %[[result]] : !tfrt_gpu.entry_point

func @main() {
  return
}
