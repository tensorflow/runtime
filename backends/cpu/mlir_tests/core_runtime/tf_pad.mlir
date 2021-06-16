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

// CHECK: --- Running 'pad_with_hardcoded_attr_f32'
func @pad_with_hardcoded_attr_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %a = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 2, 2, 1], values = [0.5 : f32] } : 1
  %padding = corert.executeop(%cpu) "tf.Const"()
    { dtype = i32, value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32> } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.Pad"(%a, %padding)
      : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 8, 8, 1], md5sum = 1170624143, values = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01, 5.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, ... ]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch)
    "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
