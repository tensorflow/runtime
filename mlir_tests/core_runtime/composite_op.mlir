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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices='cpu' | FileCheck %s --dump-input=fail

func @matmul_fn(%arg : !corert.tensorhandle) -> !corert.tensorhandle {
  %cpu = corert.get_device "cpu"
  %t1 = corert.executeop(%cpu) "tfrt_test.matmul"(%arg, %arg)
    {transpose_a = false, transpose_b = false}: 1
  hex.return %t1 : !corert.tensorhandle
}

// CHECK-LABEL: --- Running 'corert.simple_composite_op'
func @corert.simple_composite_op() -> !hex.chain {
  %cpu = corert.get_device "cpu"

  // Prepare input.
  %ch0 = hex.new.chain
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %matmul_fn_op = "corert.make_composite_op" () {fn=@matmul_fn} : () -> !corert.op

  %result = "corert.execute_crt_op" (%matmul_fn_op, %a_handle) {op_attrs =[]} : (!corert.op, !corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: shape = [1, 1], values = [4.000000e+00]
  %ch1 = "corert.print_tensorhandle"(%result, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%result, %result)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [1.600000e+01]
  %ch2 = "corert.print_tensorhandle"(%result1, %ch1) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch2 : !hex.chain
}

