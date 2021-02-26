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

// RUN: bef_executor_lite $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail

// CHECK: --- Running 'debug_info':
func @debug_info() {
  // CHECK: myNameScope0/MySimpleKernel0
  "tfrt_test.print_debug_info"() : () -> (!tfrt.chain) loc(#loc0)

  // CHECK: Kernel has no debug info
  "tfrt_test.print_debug_info"() : () -> (!tfrt.chain)

  // CHECK: myNameScope1/MySimpleKernel1
  "tfrt_test.print_debug_info"() : () -> (!tfrt.chain)
                                   loc("myNameScope1/MySimpleKernel1")

  // CHECK: foo
  "tfrt_test.print_debug_info"() : () -> (!tfrt.chain) loc(fused["foo", "bar"])

  // CHECK: bar
  %ch = "tfrt_test.print_debug_info"() : () -> (!tfrt.chain)
                                         loc(fused["foo.py":42:314, "bar"])
  // CHECK: foo/bar
  "tfrt_test.print_debug_info"(%ch) : (!tfrt.chain) -> (!tfrt.chain)
                                      loc("foo/bar")


  tfrt.return
}

// CHECK: --- Running 'debug_info_sync':
func @debug_info_sync()  attributes {tfrt.sync} {
  // CHECK: myNameScope0/MySimpleKernel0
  "tfrt_test.sync_print_debug_info"() : () -> (!tfrt.chain) loc(#loc0)

  // CHECK: Kernel has no debug info
  "tfrt_test.sync_print_debug_info"() : () -> (!tfrt.chain)

  // CHECK: myNameScope1/MySimpleKernel1
  "tfrt_test.sync_print_debug_info"() : () -> (!tfrt.chain)
                                        loc("myNameScope1/MySimpleKernel1")

  // CHECK: foo
  "tfrt_test.sync_print_debug_info"() : () -> (!tfrt.chain)
                                        loc(fused["foo", "bar"])

  // CHECK: bar
  %ch = "tfrt_test.sync_print_debug_info"() : () -> (!tfrt.chain)
                                              loc(fused["foo.py":42:314, "bar"])
  // CHECK: foo/bar
  "tfrt_test.sync_print_debug_info"(%ch) : (!tfrt.chain) -> (!tfrt.chain)
                                           loc("foo/bar")

  tfrt.return
}

#loc0 = loc("myNameScope0/MySimpleKernel0")
