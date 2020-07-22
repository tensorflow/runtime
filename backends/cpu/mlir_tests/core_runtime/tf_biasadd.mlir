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

// RUN: bef_executor -devices=cpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'biasadd_2d'
func @biasadd_2d() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[1.5, 2.5]> : tensor<2xf32> } : 1
  %output = corert.executeop(%cpu) "tf.BiasAdd"(%input_1, %input_2) { T = f32, data_format = "NHWC" } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [3, 2], values = [2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'biasadd_3d'
func @biasadd_3d() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]> : tensor<2x3x2xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[1.5, 2.5]> : tensor<2xf32> } : 1
  %output = corert.executeop(%cpu) "tf.BiasAdd"(%input_1, %input_2) { T = f32, data_format = "NHWC" } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 3, 2], values = [2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00, 2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'biasadd_4d'
func @biasadd_4d() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x2x3x2xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[1.5, 2.5]> : tensor<2xf32> } : 1
  %output = corert.executeop(%cpu) "tf.BiasAdd"(%input_1, %input_2) { T = f32, data_format = "NHWC" } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 2, 3, 2], values = [2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00, 2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'biasadd_4d_invalid_data_format'
func @biasadd_4d_invalid_data_format() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x2x3x2xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[1.5, 2.5]> : tensor<2xf32> } : 1
  // expected-error @+1 {{runtime error: invalid data format. Currently only support NHWC}}
  %output = corert.executeop(%cpu) "tf.BiasAdd"(%input_1, %input_2) { T = f32, data_format = "NCHW" } : 1

  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'biasadd_5d'
func @biasadd_5d() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]]> : tensor<1x1x2x3x2xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[1.5, 2.5]> : tensor<2xf32> } : 1
  %output = corert.executeop(%cpu) "tf.BiasAdd"(%input_1, %input_2) { T = f32, data_format = "NHWC" } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 3, 2], values = [2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00, 2.500000e+00, 4.500000e+00, 4.500000e+00, 6.500000e+00, 6.500000e+00, 8.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}
