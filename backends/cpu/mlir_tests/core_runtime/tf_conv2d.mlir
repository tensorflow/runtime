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

// CHECK: --- Running 'conv2d_valid'
func @conv2d_valid() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %conv2d_in_th1 = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%cpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NHWC",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = false}  : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [4, 2, 2, 4], md5sum = 1215977510, values = [3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, ... ]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%conv2d_th) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_valid_strides'
func @conv2d_valid_strides() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %conv2d_in_th1 = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%cpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
      {T = f32, data_format = "NHWC",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = false}  : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [4, 1, 1, 4], values = [3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%conv2d_th) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_same'
func @conv2d_same() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %conv2d_in_th1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%cpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
    { T = f32, data_format = "NHWC",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = false }  : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [4, 4, 4, 4], md5sum = 1579683316, values = [1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, ... ]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%conv2d_th) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'conv2d_same_strides'
func @conv2d_same_strides() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  %conv2d_in_th1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<4x4x4x4xf32> } : 1
  %conv2d_in_th2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<1.0> : tensor<3x3x4x4xf32> } : 1
  %conv2d_th = corert.executeop(%cpu) "tf.Conv2D"(%conv2d_in_th1, %conv2d_in_th2)
    { T = f32, data_format = "NHWC",  dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = false }  : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [4, 2, 2, 4], md5sum = 3242887444, values = [3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 2.400000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01, ... ]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%conv2d_th) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
