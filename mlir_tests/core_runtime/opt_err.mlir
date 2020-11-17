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

// RUN: tfrt_opt -split-input-file -verify-diagnostics %s

func @invalid_input(%arg : i32) { // expected-note {{prior use here}}
  %ch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch "cpu"

  // expected-error @+1 {{expects different type than prior uses: '!corert.tensorhandle' vs 'i32'}}
  %res0 = corert.executeop(%cpu) "some.op"(%arg) : 1

  tfrt.return %res0 : !corert.tensorhandle
}

// -----

func @invalid_opattrs() {
  %ch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch "cpu"

  // expected-error @+1 {{'corert.executeop' op each op_attr should be a key-value pair, where the key is a string}}
  %res0 = "corert.executeop" (%cpu)
    {op_name = "some.op", op_attrs = [1 : i32], op_func_attrs = []} : (!corert.ophandler) -> !corert.tensorhandle

  tfrt.return %res0 : !corert.tensorhandle
}
