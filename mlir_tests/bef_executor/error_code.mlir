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

// RUN: bef_executor_lite --print_error_code %s.bef 2>&1 | FileCheck %s

// CHECK-LABEL: --- Running 'test_error_result'
func @test_error_result() -> i32 {
  %x = "tfrt_test.fail"() : () -> i32 // expected-error {{something bad happened}}
  tfrt.return %x : i32
}
// CHECK-NEXT: 'test_error_result' returned <<error: something bad happened, code: Unknown>>
