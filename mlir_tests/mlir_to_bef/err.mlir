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

// RUN: tfrt_translate -mlir-to-bef -split-input-file -verify-diagnostics %s

func @function_arg() -> i32 {
  // expected-error @+1 {{all functions need to have a tfrt.return}}
  %x = "someop"() : () -> i32
}

// -----

// expected-error @+1 {{external functions are not allowed}}
func private @external_func() -> i32

// -----

func @caller() {
  %c1 = tfrt.constant.i32 1
  // expected-error @+1 {{function @missing_callee not defined}}
  "unregistered.call"(%c1) { name = @missing_callee } : (i32) -> ()
  tfrt.return
}

// -----

func @caller() {
  %c1 = tfrt.constant.i32 1
  // expected-error @+1 {{'missing_callee' does not reference a valid function}}
  tfrt.call @missing_callee(%c1) : (i32) -> ()
  tfrt.return
}

// -----

func @caller() {
  // expected-error @+1 {{BEF files cannot encode the 'type' attribute}}
  "someop"() { type = tensor<1xf32>} : () -> i32
  tfrt.return
}

// -----

func @sync_fn_return_argument(%1: i32) -> i32 attributes {tfrt.sync} {
  // expected-error @+1 {{return value 0 is an argument in a sync function}}
  tfrt.return %1 : i32
}

// -----

func @sync_fn_return_duplicated(%1: i32) -> (i32, i32) attributes {tfrt.sync} {
  %c1 = tfrt.constant.i32 1
  // expected-error @+1 {{return value 1 is duplicated in a sync function}}
  tfrt.return %c1, %c1 : i32, i32
}
