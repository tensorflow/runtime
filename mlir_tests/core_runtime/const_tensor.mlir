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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'const_dense_tensor'
func @const_dense_tensor() {
  %ch0 = tfrt.new.chain

  %a = corert.const_dense_tensor dense<[0, 1, 2]>: tensor<3xi32>

  // CHECK: shape = [3], values = [0, 1, 2]
  %ch5 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'const_string_tensor'
func @const_string_tensor() {
  %ch0 = tfrt.new.chain

  %a = corert.const_string_tensor {shape = [2], value = ["string", "tensor"]}

  // CHECK: shape = [2], values = ["string", "tensor"]
  %ch5 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'scalar_string_tensor'
func @scalar_string_tensor() {
  %ch0 = tfrt.new.chain

  %a = corert.const_string_tensor {shape = [], value = ["string"]}

  // CHECK: shape = [], values = ["string"]
  %ch5 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'create_dense_tensor'
func @create_dense_tensor() {
  %ch0 = tfrt.new.chain

  %a = corert.create_dense_tensor.ui64 {shape = [1], value = [2 : ui64]}

  // CHECK: shape = [1], values = [2]
  %ch1 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

// CHECK-LABEL: --- Running 'create_dense_tensor_bf16'
func @create_dense_tensor_bf16() {
  %ch0 = tfrt.new.chain

  %a = corert.create_dense_tensor.bf16 {shape = [1], value = [2.5 : bf16]}

  // CHECK: shape = [1], values = [Does not support printing bf16.]
  %ch1 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
