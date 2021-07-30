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

// RUN: tfrt_translate -mlir-to-bef --disable-optional-sections=true %s | tfrt_translate --bef-to-mlir --mlir-print-op-generic 2>&1 | FileCheck %s --dump-input=fail

// CHECK: warning: Missing AttributeTypes, AttributeNames or RegisterTypes sections.

// CHECK-LABEL: "builtin.func"
func @basic.constant() -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"() {{{.*}} = {{.*}}}
  // CHECK-NEXT: "tfrt.return"([[REG]]) : ({{.*}}) -> ()

  %x = "simple.op"() {value = 42 : i32} : () -> i32
  tfrt.return %x : i32
}
// CHECK: sym_name = "basic.constant"
