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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices='cpu,composite_op' | FileCheck %s --dump-input=fail

func @matmul_fn(%arg : !dht.dense_host_tensor.f32.2) -> !dht.dense_host_tensor.f32.2 {
  %cpu = corert.get_device "cpu"
  %t = "corert.ht_to_tensorhandle" (%arg) : (!dht.dense_host_tensor.f32.2) -> !corert.tensorhandle
  %t1 = corert.executeop(%cpu) "tfrt_test.matmul"(%t, %t)
    {transpose_a = 0 : i1, transpose_b = 0 : i1}: 1
  %result = "corert.tensorhandle_to_ht" (%t1) : (!corert.tensorhandle) -> !dht.dense_host_tensor.f32.2
  hex.return %result : !dht.dense_host_tensor.f32.2
}

// CHECK-LABEL: --- Running 'corert.simple_composite_op'
func @corert.simple_composite_op() -> !hex.chain {

  %fn_device = corert.get_device "composite_op"
  %cpu = corert.get_device "cpu"

  // Prepare input.
  %ch0 = hex.new.chain
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %ch1 = "corert.register_composite_op"(%fn_device) {name="matmul_fn", fn=@matmul_fn} : (!corert.device) -> !hex.chain

  %result = corert.executeop(%fn_device) "matmul_fn"(%a_handle) : 1

  // CHECK: shape = [1, 1], values = [4.000000e+00]
  %ch2 = "corert.print_tensorhandle"(%result, %ch1) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%result, %result)
    {transpose_a = 0 : i1, transpose_b = 0 : i1}: 1

  // CHECK: shape = [1, 1], values = [1.600000e+01]
  %ch3 = "corert.print_tensorhandle"(%result1, %ch2) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return %ch3 : !hex.chain
}

