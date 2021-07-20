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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu %s.bef

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

func @matmul_fn(%ch: !tfrt.chain, %arg : !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch "cpu"
  %t1 = corert.executeop(%cpu) "tfrt_test.matmul"(%arg, %arg)
    {transpose_a = false, transpose_b = false}: 1
  tfrt.return %ch, %t1 : !tfrt.chain, !corert.tensorhandle
}

// CHECK-LABEL: --- Running 'corert.simple_composite_op'
func @corert.simple_composite_op() -> !tfrt.chain {
  // Prepare input.
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %matmul_fn_op = "corert.make_composite_op" () {fn=@matmul_fn} : () -> !corert.op

  %result = "corert.execute_crt_op" (%matmul_fn_op, %a_handle) {op_attrs = [], op_func_attrs = []} : (!corert.op, !corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: shape = [1, 1], values = [4.000000e+00]
  %ch1 = "corert.print_tensorhandle"(%result, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %result1 = corert.executeop(%cpu) "tfrt_test.matmul"(%result, %result)
    {transpose_a = false, transpose_b = false}: 1

  // CHECK: shape = [1, 1], values = [1.600000e+01]
  %ch2 = "corert.print_tensorhandle"(%result1, %ch1) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

func @matmul_fn_two_results(%ch: !tfrt.chain, %arg : !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle, !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch "cpu"
  %t1 = corert.executeop(%cpu) "tfrt_test.matmul"(%arg, %arg)
    {transpose_a = false, transpose_b = false}: 1
  tfrt.return %ch, %t1, %t1: !tfrt.chain, !corert.tensorhandle, !corert.tensorhandle
}

// CHECK-LABEL: --- Running 'corert.composite_op_result_multi_fanout'
func @corert.composite_op_result_multi_fanout() -> !tfrt.chain {
  // Prepare input.
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %matmul_fn_op = "corert.make_composite_op" () {fn=@matmul_fn_two_results} : () -> !corert.op

  %result, %result_1 = "corert.execute_crt_op" (%matmul_fn_op, %a_handle) {op_attrs =[], op_func_attrs = []} : (!corert.op, !corert.tensorhandle) -> (!corert.tensorhandle, !corert.tensorhandle)

  // CHECK: shape = [1, 1], values = [4.000000e+00]
  %ch1 = "corert.print_tensorhandle"(%result, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // CHECK: shape = [1, 1], values = [4.000000e+00]
  %ch2 = "corert.print_tensorhandle"(%result_1, %ch1) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}


func @return_first(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %x : !tfrt.chain, !corert.tensorhandle
}

func @return_second(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %y : !tfrt.chain, !corert.tensorhandle
}

func @func_with_control_flow(%ch: !tfrt.chain, %arg : !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch "cpu"

  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [1 : i32] } : 1
  %b_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [2, 2], values = [2 : i32] } : 1

  %async_handle = corert.executeop(%cpu) "tfrt_test.async.noop"(%arg) : 1

  %result:2 = corert.cond %async_handle @return_first @return_second (%ch, %a_handle, %b_handle) : (!corert.tensorhandle, !corert.tensorhandle) -> (!corert.tensorhandle)

  tfrt.return %result#0, %result#1 : !tfrt.chain, !corert.tensorhandle
}


// CHECK-LABEL: --- Running 'corert.composite_op_async_output'
func @corert.composite_op_async_output() -> !tfrt.chain {
  // Prepare input.
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %true_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [1 : i32] } : 1

  %fn_op = "corert.make_composite_op" () {fn=@func_with_control_flow} : () -> !corert.op

  %result = "corert.execute_crt_op" (%fn_op, %true_handle) {op_attrs =[], op_func_attrs = []} : (!corert.op, !corert.tensorhandle) -> (!corert.tensorhandle)

  // CHECK: DenseHostTensor dtype = i32, shape = [2, 2], values = [1, 1, 1, 1]
  %ch1 = "corert.print_tensorhandle"(%result, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}
