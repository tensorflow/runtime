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

// RUN: bef_executor_lite $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail --dump-input-filter=all

// CHECK: --- Running 'debug_info':
func @debug_info() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  // CHECK: myNameScope0/MySimpleKernel0
  %ch1 = "tfrt_test.print_debug_info"(%ch0) : (!tfrt.chain) -> (!tfrt.chain) loc(#loc0)

  // CHECK: Kernel has no debug info
  %ch2 = "tfrt_test.print_debug_info"(%ch1) : (!tfrt.chain) -> (!tfrt.chain)

  // CHECK: myNameScope1/MySimpleKernel1
  %ch3 = "tfrt_test.print_debug_info"(%ch2) : (!tfrt.chain) -> (!tfrt.chain)
                                   loc("myNameScope1/MySimpleKernel1")

  // CHECK: foo
  %ch4 = "tfrt_test.print_debug_info"(%ch3) : (!tfrt.chain) -> (!tfrt.chain) loc(fused["foo", "bar"])

  // CHECK: bar
  %ch5 = "tfrt_test.print_debug_info"(%ch4) : (!tfrt.chain) -> (!tfrt.chain)
                                         loc(fused["foo.py":42:314, "bar"])
  // CHECK: foo/bar
  %ch6 = "tfrt_test.print_debug_info"(%ch5) : (!tfrt.chain) -> (!tfrt.chain)
                                      loc("foo/bar")

  // CHECK: foo
  %ch7 = "tfrt_test.print_debug_info"(%ch6) : (!tfrt.chain) -> (!tfrt.chain)
                                   loc(callsite("foo" at "bar.py":42:314))

  tfrt.return %ch7 : !tfrt.chain
}

#loc0 = loc("myNameScope0/MySimpleKernel0")
