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

// RUN: bef_executor_lite --test_init_function __init__ --functions print_hello,print_bye %s.bef 2>&1 | FileCheck %s --dump-input=fail

// CHECK: --- Running '__init__'
// CHECK: string = initializing!

// CHECK: --- Running 'print_hello'
func @print_hello() {
  // CHECK: hello host executor!
  %ch0 = tfrt.new.chain
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return
}

// CHECK: --- Running 'print_bye'
func @print_bye() {
  %ch0 = tfrt.new.chain
  %bye = "tfrt_test.get_string"() { value = "bye host executor!" } : () -> !tfrt.string

  // CHECK: string = bye host executor!
  %ch1 = "tfrt_test.print_string"(%bye, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)
  tfrt.return
}

func @__init__() {
  %ch0 = tfrt.new.chain
  %bye = "tfrt_test.get_string"() { value = "initializing!" } : () -> !tfrt.string
  %ch1 = "tfrt_test.print_string"(%bye, %ch0) : (!tfrt.string, !tfrt.chain) -> (!tfrt.chain)
  tfrt.return
}
