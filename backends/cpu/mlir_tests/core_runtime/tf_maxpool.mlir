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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu %s.bef | FileCheck %s

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'maxpool_valid_padding'
func @maxpool_valid_padding() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %cpu_handle_input = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [1, 3, 3, 1], values = [0.0 : f32, 1.0 : f32, 2.0 : f32,  3.0 : f32,  4.0 : f32,  5.0 : f32,  6.0 : f32,  7.0 : f32,  8.0 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.MaxPool"(%cpu_handle_input)
    { ksize = [1, 3, 3, 1], padding = "VALID", strides = [1, 2, 2, 1], data_format="NHWC" } : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 1, 1, 1], values = [8.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'maxpool_same_padding'
func @maxpool_same_padding() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %maxpool_in_th = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<1x3x3x1xf32> } : 1
  %cpu_handle_result = corert.executeop(%cpu) "tf.MaxPool"(%maxpool_in_th)
    { T = f32, data_format = "NHWC",  ksize = [1, 3, 3, 1], padding = "SAME", strides = [1, 2, 2, 1]} : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 2, 2, 1], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
