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

// RUN: tfrt_opt %s -split-input-file --verify-diagnostics

func @bad_return(%x: i32) {

  // expected-error @+1 {{'tfrt.return' op has 1 operands, but enclosing function returns 0}}
  tfrt.return %x : i32
}

// -----

func @return_mismatch(%x: i32) -> !tfrt.chain {

  // expected-error @+1 {{type of return operand 0 ('i32') doesn't match function result type ('!tfrt.chain')}}
  tfrt.return %x : i32
}

// -----

func @if_mismatch(%cond: i1, %v1: i32, %v2: i32) -> i32 {

  tfrt.if %cond, %v1 : (i32) -> () {

    // expected-error @+1 {{'tfrt.return' op operand types don't match 'tfrt.if' result}}
    tfrt.return %v1 : i32
  }
  tfrt.return %v1 : i32
}
