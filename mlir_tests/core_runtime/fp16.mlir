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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu $(bef_name %s) | FileCheck %s --dump-input=fail

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK-LABEL: --- Running 'cast_f16'
func @cast_f16() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // Create tensor whose shape is represented using RepKind::kRep32.
  %float_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [-1.0 : f32] } : 1

  %half_handle = corert.executeop(%cpu)
    "tfrt_test.cast"(%float_handle) { type = "f16" } : 1

  %half_result = corert.executeop(%cpu)
    "tfrt_test.relu"(%half_handle) : 1

  // CHECK: DenseHostTensor dtype = f16, shape = [1, 1], values = [fp16(0)]
  %ch1 = "corert.print_tensorhandle"(%half_result, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %float_result = corert.executeop(%cpu)
    "tfrt_test.cast"(%half_result) { type = "f32" } : 1

  // CHECK: shape = [1, 1], values = [0.000000e+00]
  %ch2 = "corert.print_tensorhandle"(%float_result, %ch1) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}
