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

// RUN: cat %s.bef | not bef_executor_lite | FileCheck %s

// CHECK: --- Running 'test_leak_one_int32'
func @test_leak_one_int32() -> () {
  "tfrt_test.memory_leak_one_int32"() : () -> ()
  tfrt.return
}
// CHECK: Memory leak detected: 1 alive allocations, 4 alive bytes
