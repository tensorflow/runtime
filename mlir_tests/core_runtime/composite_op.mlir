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

func.func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

func.func @return_first(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %x : !tfrt.chain, !corert.tensorhandle
}

func.func @return_second(%in: !tfrt.chain, %x: !corert.tensorhandle, %y: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  tfrt.return %in, %y : !tfrt.chain, !corert.tensorhandle
}

func.func @func_with_control_flow(%ch: !tfrt.chain, %arg : !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
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
func.func @corert.composite_op_async_output() -> !tfrt.chain {
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
