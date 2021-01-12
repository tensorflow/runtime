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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK-LABEL: Running 'typed_attr'
func @typed_attr() -> !tfrt.chain {
  %ch = tfrt.new.chain
  // CHECK: I64: 1
  // CHECK-NEXT: F32: 5.000000e-01
  // CHECK-NEXT: I1: 0
  %ch0 = tfrt_test.print_typed_attr %ch {value = 1 : i64}
  %ch1 = tfrt_test.print_typed_attr %ch0 {value = 0.5 : f32}
  %ch2 = tfrt_test.print_typed_attr %ch1 {value = false}
  tfrt.return %ch2 : !tfrt.chain
}
