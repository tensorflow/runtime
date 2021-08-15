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

// RUN: bef_executor_lite %s.bef 2>&1 | FileCheck %s

// CHECK: --- Running 'unique_loc_test'
func @unique_loc_test() -> !tfrt.chain {
  %ch0 = "tfrt_test.unique_loc"() {name = "a"} : () -> !tfrt.chain loc(unknown)
  %ch1 = "tfrt_test.unique_loc"() {name = "b"} : () -> !tfrt.chain loc(unknown)
  %ch2 = "tfrt_test.unique_loc"() {name = "c"} : () -> !tfrt.chain loc(unknown)
  %ch = tfrt.merge.chains %ch0, %ch1, %ch2 : !tfrt.chain, !tfrt.chain, !tfrt.chain
  // CHECK-NOT: returned <<error
  tfrt.return %ch : !tfrt.chain
}
