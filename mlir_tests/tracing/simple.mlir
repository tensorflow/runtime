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

// RUN: bef_executor_debug_tracing --enable_tracing --tracing_level=debug $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail

// CHECK: --- Running 'print_test'
func @print_test() {
// CHECK: Scope:Bef Executor
// CHECK: Scope:Function: print_test
// CHECK: Scope:BEFExecutor::ProcessReadyKernels

  // CHECK: Scope:tfrt.new.chain
  %ch0 = tfrt.new.chain
  // CHECK: End Scope

  // CHECK: Scope:tfrt_test.print_hello
  %ch1 = "tfrt_test.print_hello"(%ch0) : (!tfrt.chain) -> !tfrt.chain
  // CHECK: End Scope

  tfrt.return

// CHECK: End Scope
// CHECK: End Scope
// CHECK: End Scope
}